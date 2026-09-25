import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.domain import Rules, Track, Yard, Action
from fzd_shunting.environment import Environment
from fzd_shunting.learning import train, update_policy
from fzd_shunting.planning import replay
from fzd_shunting.policy.gnn import (
    GraphPolicy,
    collate_graphs,
    encode_graph,
    load_model,
)
from fzd_shunting.request import load_request
from fzd_shunting.sampling import ParallelSampler


def scenes():
    yard = Yard(
        (
            Track("R", 0, 0, "transit"),
            Track("A", 100000),
            Track("B", 100000),
            Track("C", 100000),
        ),
        (("A", "R"), ("B", "R"), ("C", "R")),
    )
    result = []
    for n in (1, 3):
        rows = [
            {
                "No": f"v{i}",
                "Line": "A",
                "Position": i + 1,
                "Length": 13.2,
                "TargetLines": {
                    "B" if i % 2 else "C": {"ForceTargetPosition": [2 + i, 6 + i]}
                },
            }
            for i in range(n)
        ]
        result.append(
            Dispatcher(
                Environment(
                    yard,
                    load_request(
                        {"StartStatus": rows, "locoNode": {"Line": "R"}}, yard
                    ),
                )
            )
        )
    return result


def policy(dispatcher, device="cpu"):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model = GraphPolicy(hidden=32, layers=2).to(device)
    model.bind(dispatcher)
    return model


def test_disjoint_batch_matches_independent_logits_and_gradients():
    a, b = scenes()
    m = policy(a)
    graphs = [encode_graph(d, d.state, d.candidates(d.state)) for d in (a, b)]
    separate = [
        m.forward_batch(collate_graphs([g]))[0, : len(g.action_features)]
        for g in graphs
    ]
    combined = m.forward_batch(collate_graphs(graphs))
    for i, g in enumerate(graphs):
        torch.testing.assert_close(
            combined[i, : len(g.action_features)], separate[i], atol=2e-6, rtol=2e-5
        )
        assert torch.isneginf(combined[i, len(g.action_features) :]).all()
    sum(x.square().sum() for x in separate).backward()
    expected = {
        n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None
    }
    m.zero_grad()
    combined[torch.isfinite(combined)].square().sum().backward()
    for n, p in m.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=3e-6, rtol=5e-4)


def test_parallel_cpu_actors_preserve_greedy_trajectory_and_replay():
    ds = scenes()
    m = policy(ds[0])
    out = []
    for workers in (0, 2):
        with ParallelSampler(ds, workers=workers) as sampler:
            traces, timing = sampler.collect(m, [0, 1], 2, 12, greedy=True)
            out.append([t.outcome.actions for t in traces])
            assert all(t.outcome is not None for t in traces)
            assert timing["environment_steps"] == sum(
                len(t.outcome.actions) for t in traces
            )
            for trajectory in traces:
                d = ds[trajectory.request_index]
                final = replay(d, trajectory.outcome.actions)
                assert final == trajectory.outcome.final_state
    assert out[0] == out[1]


def test_policy_update_reuses_graphs_without_environment_or_teacher(
    tmp_path, monkeypatch
):
    d = scenes()[1]
    m = policy(d)
    with ParallelSampler([d], workers=0) as sampler:
        traces, _ = sampler.collect(m, [0, 0], 3, 8)
    import random

    def forbidden(*args, **kw):
        raise AssertionError("environment recomputed during update")

    monkeypatch.setattr(Dispatcher, "candidates", forbidden)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0)
    result = update_policy(
        m,
        optimizer,
        traces,
        group_size=3,
        update_epochs=2,
        minibatch_size=5,
        clip_ratio=0.2,
        entropy_weight=0.01,
        target_kl=0.03,
        precision="float32",
        rng=random.Random(1),
    )
    assert result["optimizer_updates"] > 0
    assert all(abs(sum(g["advantages"])) < 1e-5 for g in result["groups"])
    assert not hasattr(m, "critic")


def test_old_training_contract_removed_and_periodic_resume(tmp_path):
    d = scenes()[0]
    options = dict(
        iterations=2,
        max_hooks=6,
        group_size=2,
        groups_per_batch=1,
        workers=0,
        device="cpu",
        precision="float32",
        hidden=32,
        layers=2,
        update_epochs=1,
        checkpoint_every=1,
        output=tmp_path / "policy.pt",
    )
    result = train(d, **options)
    cp = torch.load(options["output"], weights_only=True)
    assert (
        cp["schema_version"] == 4 and cp["algorithm"] == "grouped-root-fork-relative-v2"
    )
    assert not {"teacher_summary", "reference_state", "synthetic_requests"} & cp.keys()
    assert result["iterations"] == 2 and cp["iterations_completed"] == 2
    assert (
        len(
            Path(options["output"])
            .with_suffix(".history.jsonl")
            .read_text()
            .splitlines()
        )
        == 2
    )
    history = Path(options["output"]).with_suffix(".history.jsonl")
    with history.open("a") as stream:
        stream.write('{"iteration": 3, "uncommitted":')
    result = train(d, **{**options, "iterations": 1}, resume=options["output"])
    assert result["iterations"] == 3
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [1, 2, 3]
    assert not any("uncommitted" in row for row in rows)
    with pytest.raises(TypeError):
        train(d, **options, warmup_epochs=1)
    path = tmp_path / "old.pt"
    torch.save({"schema_version": 2}, path)
    with pytest.raises(ValueError, match="schema 4"):
        load_model(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batched_update_and_model_reload(tmp_path):
    a, b = scenes()
    m = policy(a, "cuda")
    graphs = [encode_graph(d, d.state, d.candidates(d.state)) for d in (a, b)]
    batch = collate_graphs(graphs).pin_memory().to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m.forward_batch(batch)
        loss = logits[batch.action_mask].square().sum()
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )
    result = train(
        [a, b],
        iterations=1,
        max_hooks=6,
        group_size=2,
        groups_per_batch=2,
        workers=0,
        device="cuda",
        precision="bfloat16",
        hidden=32,
        layers=2,
        minibatch_size=4,
        output=tmp_path / "gpu.pt",
    )
    assert result["runtime"]["device"] == "cuda"
    restored = load_model(tmp_path / "gpu.pt", "cuda")
    c = a.candidates(a.env.scenario.initial)
    assert restored.choose(a, a.env.scenario.initial, c) in c
