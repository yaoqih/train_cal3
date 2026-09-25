import json
import random
from dataclasses import replace
from collections import defaultdict

import pytest
import numpy as np

torch = pytest.importorskip("torch")
from test_learning_contract import make, car
from fzd_shunting.domain import Action, ActionLimit, BusinessContext, Rules, Track, Yard
from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.environment import Environment
from fzd_shunting.policy.gnn import GraphPolicy, encode_graph, collate_graphs
from fzd_shunting.sampling import (
    Actor,
    ScenarioSpec,
    StartPoint,
    StatePool,
    ParallelSampler,
)
from fzd_shunting.learning import update_policy, evaluate, split_dataset, train
from fzd_shunting.objective import trajectory_score
from fzd_shunting.frontier import FrontierEvaluator
from fzd_shunting.planning import SearchNode, next_hook, plan, replay
from fzd_shunting.feasibility import terminal_feasibility
from fzd_shunting.decisions import DecisionCache


def model(d):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    m = GraphPolicy(hidden=32, layers=2, hook_budget=12)
    m.bind(d)
    return m


def branch_scene():
    d = make(
        [
            car("a", "A"),
            car(
                "guard", "修1库内", 3, target={"修1库内": {"ForceTargetPosition": [3]}}
            ),
        ],
        rules=Rules(action_limits=(ActionLimit("once", ("C",), 1),)),
    )
    actions = (Action("A", "get", 1), Action("C", "put", 1))
    state = replay(d, actions)
    return d, StartPoint(0, state, d.context, actions)


def test_joint_distribution_balances_line_operation_groups():
    d = make([car("a", "A", 1), car("b", "A", 2), car("c", "A", 3), car("d", "C")])
    m = model(d)
    for p in m.actor.parameters():
        p.data.zero_()
    candidates = d.candidates(d.state)
    probabilities = m(d, d.state, candidates).softmax(0).detach().numpy()
    grouped = defaultdict(list)
    for c, p in zip(candidates, probabilities):
        grouped[(c.action.line, c.action.operation)].append(p)
    assert sorted(map(len, grouped.values())) == [1, 3]
    for ps in grouped.values():
        np.testing.assert_allclose(sum(ps), 0.5, atol=1e-6)
        np.testing.assert_allclose(ps, [0.5 / len(ps)] * len(ps), atol=1e-6)


def test_source_destination_boundaries_and_counterfactual_order():
    d = make([car("a", "A", 1), car("b", "A", 2), car("c", "C"), car("p", "机车")])
    c = d.prepare(d.state, Action("A", "get", 1))
    g = encode_graph(d, d.state, (c,), hook_budget=12)
    e = d._graph_encoder
    assert not np.array_equal(
        g.nodes[:, 28], encode_graph(d, d.state, (c,), hook_budget=24).nodes[:, 28]
    )
    assert tuple(g.action_nodes[0]) == (
        e.tracks["A"],
        e.loco,
        e.vehicles["a"],
        e.vehicles["a"],
        e.vehicles["b"],
        e.vehicles["p"],
    )
    for index, expected in zip(g.action_posts[0], [("b",), ("p", "a")]):
        actual = g.post_vehicles[g.post_owners == index]
        assert actual.tolist() == [e.vehicles[x] for x in expected]
    put = d.prepare(c.after, Action("C", "put", 1), c.after_context)
    pg = encode_graph(d, c.after, (put,), c.after_context, 12)
    assert tuple(pg.action_nodes[0]) == (
        e.loco,
        e.tracks["C"],
        e.vehicles["a"],
        e.vehicles["a"],
        e.vehicles["p"],
        e.vehicles["c"],
    )
    index = pg.action_posts[0, 1]
    assert pg.post_vehicles[pg.post_owners == index].tolist() == [
        e.vehicles["a"],
        e.vehicles["c"],
    ]
    later = encode_graph(d, c.after, (put,), c.after_context, 24)
    assert not np.array_equal(pg.nodes[:, 28], later.nodes[:, 28])


def test_failed_short_trajectories_do_not_get_hook_bonus():
    d = make()
    score = trajectory_score(d, d.state, d.context, 64, "hook_budget")
    assert score == trajectory_score(
        d, replace(d.state, hook=40), d.context, 64, "hook_budget"
    )
    assert trajectory_score(d, d.state, d.context, 64, "cycle") < score
    end = replay(d, (Action("A", "get", 1), Action("B", "put", 1)))
    assert (
        trajectory_score(d, end, d.context, 64)
        > trajectory_score(d, replace(end, hook=40), d.context, 64)
        > score
    )


def test_next_hook_enforces_the_whole_plan_budget_without_committing():
    d = make()
    m = model(d)
    d.step(d.state, d.prepare(d.state, Action("A", "get", 1)))
    before = d.snapshot()
    result = next_hook(d, m, max_hooks=1)
    assert result["status"] == "hook_budget" and result["action"] is None
    assert d.snapshot() == before
    assert next_hook(d, m, max_hooks=2)["status"] == "action"
    d.step(d.state, d.prepare(d.state, Action("B", "put", 1)))
    assert next_hook(d, m, max_hooks=2)["status"] == "complete"
    with pytest.raises(ValueError, match="below current hook"):
        next_hook(d, m, max_hooks=1)


def test_branch_prefix_protection_quota_and_total_hooks_survive_parallel_sampling():
    d, start = branch_scene()
    m = model(d)
    for workers in (0, 2):
        with ParallelSampler([d], workers=workers) as sampler:
            traces, timing = sampler.collect(m, [0], 2, 10, starts=[start])
        for t in traces:
            assert t.start_hook == 2
            assert len(t.samples) == 1
            assert t.outcome.actions[:2] == start.prefix
            assert len(t.outcome.actions) == t.outcome.final_state.hook <= 10
            assert replay(d, t.outcome.actions) == t.outcome.final_state
            assert t.outcome.final_context.count("once") == 1
            assert t.outcome.final_state.stack("修1库内") == ("guard",)
    actor = Actor([ScenarioSpec.from_dispatcher(d)])
    actor.reset([(0, start)], 10, True)
    with pytest.raises(ValueError, match="legal prefix"):
        actor.reset([(0, replace(start, state=replace(start.state, hook=3)))], 10, True)
    restored = StartPoint.from_dict(json.loads(json.dumps(start.to_dict())))
    assert restored == start
    pool = StatePool(4)
    pool.add([start], [d], 10)
    with pytest.raises(ValueError, match="validation/test"):
        pool.add([replace(start, request_index=1)], [d], 10)


def test_complete_groups_control_optimizer_steps_and_forks_credit_first_action():
    d = make()
    m = model(d)
    with ParallelSampler([d], workers=0) as sampler:
        traces, _ = sampler.collect(m, [0] * 8, 2, 10)
    kw = dict(
        group_size=2,
        update_epochs=2,
        minibatch_size=4,
        clip_ratio=0.2,
        entropy_weight=0.01,
        target_kl=0,
        precision="float32",
        rng=random.Random(1),
        trajectory_batch_size=4,
    )
    result = update_policy(m, torch.optim.AdamW(m.parameters(), lr=1e-4), traces, **kw)
    assert result["optimizer_updates"] == 8
    d, start = branch_scene()
    m = model(d)
    with ParallelSampler([d], workers=0) as sampler:
        traces, _ = sampler.collect(m, [0], 2, 10, starts=[start])
    result = update_policy(m, torch.optim.AdamW(m.parameters(), lr=1e-4), traces, **kw)
    assert result["decision_samples"] == 2
    traces[1].start_key = "different-state"
    with pytest.raises(ValueError, match="same-state"):
        update_policy(m, torch.optim.AdamW(m.parameters()), traces, **kw)


def test_exact_cache_keys_include_hook_context_and_budget_and_inference_is_frozen_per_collect():
    d = make()
    cache = DecisionCache()
    a = cache.get(d, d.state, d.context, 10)
    assert cache.get(d, d.state, d.context, 10) is a
    b = cache.get(d, replace(d.state, hook=1), d.context, 10)
    c = cache.get(d, d.state, d.context, 12)
    assert a[2] != b[2] and a[2] != c[2]
    assert cache.hits == 1
    m = model(d)
    with ParallelSampler([d], workers=0) as sampler:
        for _ in range(2):
            _, timing = sampler.collect(m, [0], 8, 8, greedy=True)
            assert timing["inference_cache_hits"] > 0
            assert 0 < timing["inference_samples"] < timing["environment_steps"]


def test_evaluation_restores_rng_and_reports_greedy_and_sampled_40_hook_success():
    d = make()
    m = model(d)
    before = torch.get_rng_state().clone()
    report = evaluate(m, [d], max_hooks=8, samples_per_request=4)
    assert torch.equal(before, torch.get_rng_state())
    assert report["sampled"]["samples_per_request"] == 4
    assert report["complete_within_40"] == report["complete"]


def test_terminal_relaxations_find_impossible_requests_without_claiming_solvable():
    d = make(
        [
            car("a", "A", 1, {"B": {"ForceTargetPosition": [1]}}),
            car("b", "A", 2, {"B": {"ForceTargetPosition": [1]}}),
        ]
    )
    assert terminal_feasibility(d)["status"] == "infeasible"
    yard = Yard(
        (Track("R", 0, 0, "transit"), Track("A", 80000), Track("B", 20000)),
        (("A", "R"), ("B", "R")),
    )
    d = make([car("a", "A", 1), car("b", "A", 2)], yard=yard)
    assert "TARGET_LENGTH_ASSIGNMENT_RELAXATION" in terminal_feasibility(d)["reasons"]
    assert terminal_feasibility(make())["status"] == "not_disproved"
    assert terminal_feasibility(make([]))["status"] == "not_disproved"


def test_explicit_invalid_sampling_runtime_does_not_fall_back():
    d = make()
    with ParallelSampler([d], workers=0) as sampler:
        with pytest.raises(ValueError, match="supported CUDA"):
            sampler.collect(model(d), [0], 2, 8, precision="bfloat16")
        with pytest.raises(ValueError, match="supported precisions"):
            sampler.collect(model(d), [0], 2, 8, precision="float16")


def test_date_partition_reserves_a_test_set():
    ds = []
    for i, day in enumerate(["20260105", "20260112", "20260119", "20260126"]):
        d = make([car(str(i), "A")])
        ds.append(
            Dispatcher(Environment(d.env.yard, replace(d.env.scenario, name=day)))
        )
    train_ds, val, test, split = split_dataset(ds, 0.25, 0.25)
    assert (len(train_ds), len(val), len(test)) == (2, 1, 1)
    assert not set(split["training_families"]) & set(split["test_families"])
    assert test[0].env.scenario.name == "20260126"


def test_parallel_frontier_matches_serial_and_plan_replays():
    d = make()
    m = model(d).eval()
    c = d.prepare(d.state, Action("A", "get", 1))
    nodes = [SearchNode(d.state, d.context), SearchNode(c.after, c.after_context)]
    outputs = []
    for workers in (0, 2):
        evaluator = FrontierEvaluator(d, m, 10, workers)
        try:
            rows = evaluator.rank(nodes)
            outputs.append([[(c.action, score) for c, score in row] for row in rows])
            evaluator.rank(nodes)
            assert evaluator.stats["inference_cache_hits"] == 2
        finally:
            evaluator.close()
    for a, b in zip(*outputs):
        assert [x[0] for x in a] == [x[0] for x in b]
        np.testing.assert_allclose([x[1] for x in a], [x[1] for x in b], atol=1e-6)
    result = plan(d, policy=m, max_hooks=10, max_expansions=80, inference_batch_size=4)
    assert result["status"] == "complete" and result["total_hooks"] == 2
    assert d.done(replay(d, result["actions"], result["events"]))


def test_branch_pool_is_checkpointed_and_restored(tmp_path):
    d = make([car("a", "A", 1), car("b", "A", 2), car("c", "C")])
    opts = dict(
        iterations=3,
        max_hooks=12,
        group_size=2,
        groups_per_batch=2,
        workers=0,
        device="cpu",
        precision="float32",
        hidden=32,
        layers=2,
        update_epochs=1,
        branch_fraction=0.5,
        trajectory_batch_size=4,
        output=tmp_path / "p.pt",
    )
    train(d, **opts)
    cp = torch.load(opts["output"], weights_only=True)
    assert cp["state_pool"]
    rows = [
        json.loads(x)
        for x in opts["output"].with_suffix(".history.jsonl").read_text().splitlines()
    ]
    assert any(r["branch_trajectories"] for r in rows[1:])
    result = train(d, **{**opts, "iterations": 1}, resume=opts["output"])
    assert result["iterations"] == 4
