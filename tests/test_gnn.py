import pytest

torch = pytest.importorskip("torch")

from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.domain import Track, Yard
from fzd_shunting.environment import Environment
from fzd_shunting.policy.gnn import GraphPolicy, load_model
from fzd_shunting.learning import train
from fzd_shunting.planning import rollout, replay
from fzd_shunting.request import load_request


def test_graph_scoring_and_real_gradient_update(tmp_path):
    torch.set_num_threads(1)
    yard = Yard(
        (Track("R", 0, 0, "transit"), Track("A", 50000, 4), Track("B", 50000, 4)),
        (("A", "R"), ("B", "R")),
    )
    request = {
        "StartStatus": [
            {
                "No": "a",
                "Line": "A",
                "Position": 1,
                "Length": 13.2,
                "TargetLines": {"B": {"ForceTargetPosition": [2, 3]}},
            },
        ],
        "locoNode": {"Line": "R"},
    }
    d = Dispatcher(Environment(yard, load_request(request, yard)))
    candidates = d.candidates(d.env.scenario.initial)
    model = GraphPolicy()
    logits = model(d, d.env.scenario.initial, candidates)
    assert logits.shape == (len(candidates),) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    result = train(
        d,
        iterations=2,
        max_hooks=6,
        output=tmp_path / "model.pt",
        group_size=2,
        update_epochs=1,
        workers=0,
        device="cpu",
        precision="float32",
        groups_per_batch=1,
    )
    assert result["updates"] == 2
    assert (tmp_path / "model.metrics.json").exists()
    restored = load_model(tmp_path / "model.pt")
    result = rollout(d, restored, max_hooks=6)
    state = replay(d, result["actions"], result["events"])
    assert (result["status"] == "complete") == d.done(state)
