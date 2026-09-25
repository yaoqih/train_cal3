import json
from dataclasses import replace
from pathlib import Path

import pytest

from fzd_shunting.domain import (
    Action,
    ActionLimit,
    BusinessContext,
    InvalidAction,
    Rules,
    State,
    Track,
    Yard,
)
from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.environment import Environment
from fzd_shunting.planning import next_hook, plan, replay, rollout
from fzd_shunting.request import load_request
from fzd_shunting.data import slot_witness


def make(rows=None, rules=Rules(), yard=None):
    yard = yard or Yard(
        (
            Track("R", 0, 0, "transit"),
            Track("A", 80000, 6),
            Track("B", 80000, 6),
            Track("C", 80000, 6),
            Track("修1库内", 151700, 7),
        ),
        (("A", "R"), ("B", "R"), ("C", "R"), ("修1库内", "R")),
    )
    rows = rows if rows is not None else [car("a", "A", target="B")]
    return Dispatcher(
        Environment(
            yard, load_request({"StartStatus": rows, "locoNode": {"Line": "R"}}, yard)
        ),
        rules,
    )


def car(no, line, pos=1, target="B", **extra):
    return {
        "No": no,
        "Line": line,
        "Position": pos,
        "Length": 13.2,
        "TargetLines": {target: {}} if isinstance(target, str) else target,
        **extra,
    }


def test_only_one_target_position_schema():
    with pytest.raises(ValueError, match="inside TargetLines"):
        make([car("a", "修1库内", 2, target="修1库内", ForceTargetPosition=[2])])
    with pytest.raises(ValueError, match="mapping"):
        make([{**car("a", "A"), "TargetLines": ["B"]}])
    d = make([car("a", "修1库内", 2, target={"修1库内": {"ForceTargetPosition": [2]}})])
    assert d.protected == {"a"}
    with pytest.raises(InvalidAction, match="PROTECTED"):
        d.prepare(d.state, Action("修1库内", "get", 1))


@pytest.mark.parametrize(
    "extra", [{"ForceTargetLine": "B"}, {"forcetrargetline": "B"}, {"IsHeavy": "false"}]
)
def test_unknown_vehicle_rules_and_non_boolean_flags_rejected(extra):
    with pytest.raises(ValueError):
        make([car("a", "A", **extra)])


def test_state_and_business_serialization_have_one_shape():
    d = make()
    state = d.state.to_dict()
    state["stacks"] = list(state["stacks"].items())
    with pytest.raises(ValueError, match="mapping"):
        State.from_dict(state)
    with pytest.raises(ValueError, match="counters"):
        BusinessContext.from_dict({"old_counts": {}})
    raw = {"StartStatus": [car("a", "机车")], "locoNode": {"Line": "R", "Vehicles": []}}
    with pytest.raises(ValueError, match="must agree"):
        load_request(raw, d.env.yard)


def test_unknown_access_rule_is_not_ignored(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {"access_rules": [{"lines": ["A"], "earliest_hook": 1, "close_after": 4}]}
        )
    )
    with pytest.raises(ValueError, match="unsupported access"):
        Rules.load(path)


def test_replay_checks_final_snapshot(tmp_path):
    from fzd_shunting.cli import main, make_dispatcher

    root = Path(__file__).resolve().parents[1]
    request = str(root / "scenarios/demo.json")
    rules = str(root / "configs/dispatch.json")
    d = make_dispatcher(request, Yard.load(), Rules.load(rules))
    result = plan(d, max_expansions=20, anytime=False)
    assert result["status"] == "complete"
    result["final_snapshot"]["state"]["hook"] += 1
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="final snapshot"):
        main(["replay", request, str(path), "--rules", rules])


def test_custom_yard_limits_loaded_unknown_models_not_substituted(tmp_path):
    data = {
        "tracks": [{"name": "A", "length_mm": 100000, "kind": "storage"}],
        "edges": [],
        "locomotive_mm": 20000,
        "train_limit_mm": 50000,
        "train_count_limit": 3,
        "route_model": "another-model",
    }
    path = tmp_path / "yard.json"
    path.write_text(json.dumps(data))
    y = Yard.load(path)
    assert (y.locomotive_mm, y.train_limit_mm, y.train_count_limit) == (20000, 50000, 3)
    scenario = load_request(
        {"StartStatus": [car("a", "A", target="A")], "locoNode": {"Line": "A"}}, y
    )
    with pytest.raises(ValueError, match="no registered router"):
        Environment(y, scenario)

    class Router:
        model = "custom-audit-v1"
        capabilities = {"verified": [], "not_modelled": ["audit_router"]}

        def route(self, state, line):
            return (line,)

    d = Dispatcher(Environment(y, scenario, Router()))
    d.step(d.state, d.prepare(d.state, Action("A", "get", 1)))
    assert d.events[-1]["route_model"] == "custom-audit-v1"
    assert d.profile["route_model"] == "custom-audit-v1"


def test_strict_goal_and_state_invariants_and_duplicate_commit():
    d = make()
    state = d.state
    empty = replace(state, stacks=tuple((line, ()) for line, _ in state.stacks))
    assert not d.done(empty)
    assert d.validate(replace(state, stacks=state.stacks + (("C", ()),)))
    assert d.validate(replace(state, hook=-1))
    candidate = d.prepare(state, Action("A", "get", 1))
    d.step(state, candidate)
    with pytest.raises(InvalidAction, match="STALE"):
        d.step(state, candidate)
    assert len(d.events) == 1


def test_context_branches_are_pure_and_part_of_candidate_identity():
    rules = Rules(action_limits=(ActionLimit("once", ("B",), 1),))
    d = make(rules=rules)
    first = d.prepare(d.state, Action("A", "get", 1))
    put = d.prepare(first.after, Action("B", "put", 1), first.after_context)
    assert put.after_context.count("once") == 1
    assert d.context.count("once") == 0 and d.events == []
    get = d.prepare(put.after, Action("B", "get", 1), put.after_context)
    with pytest.raises(InvalidAction, match="ACTION_LIMIT"):
        d.prepare(get.after, Action("B", "put", 1), get.after_context)
    different = BusinessContext()
    assert d.search_key(get.after, different) != d.search_key(
        get.after, get.after_context
    )
    assert d.decision_key(get.after, different) != d.decision_key(
        get.after, get.after_context
    )
    assert d.prepare(get.after, Action("B", "put", 1), different)


def test_snapshot_preserves_request_rules_and_history():
    d = make(rules=Rules(action_limits=(ActionLimit("once", ("B",), 1),)))
    d.step(d.state, d.prepare(d.state, Action("A", "get", 1)))
    d.step(d.state, d.prepare(d.state, Action("B", "put", 1)))
    snapshot = json.loads(json.dumps(d.snapshot()))
    other = make(rules=d.rules)
    other.restore(snapshot)
    assert other.state == d.state and other.context == d.context
    with pytest.raises(ValueError, match="mismatch"):
        make().restore(snapshot)
    altered = make(rows=[car("a", "A", target="C")], rules=d.rules)
    with pytest.raises(ValueError, match="mismatch"):
        altered.restore(snapshot)


def test_protected_inner_redelivery_is_allowed_not_locked():
    d = make(
        [
            car(
                "guard", "修1库内", 3, target={"修1库内": {"ForceTargetPosition": [3]}}
            ),
            car(
                "a",
                "机车",
                target={"修1库内": {"ForceTargetPosition": [1, 2]}, "B": {}},
            ),
        ]
    )
    for a in [
        Action("修1库内", "put", 1),
        Action("修1库内", "get", 1),
        Action("B", "put", 1),
    ]:
        d.step(d.state, d.prepare(d.state, a))
    assert d.done(d.state)


def test_goal_augmentation_share_relative_domain():
    import random

    d = make([car("a", "B", target={"B": {"ForceTargetPosition": [99]}})])
    assert d.done(d.state)
    assert slot_witness(d.env.scenario, d.env.yard, random.Random(0)) == {
        "a": ("B", 99)
    }


def test_search_policy_is_used_and_budgeted_and_plan_replay_is_versioned():
    from fzd_shunting.policy.rule import RulePolicy

    class Observed(RulePolicy):
        calls = 0

        def rank(self, d, candidates):
            self.calls += 1
            return super().rank(d, candidates)

    policy = Observed()
    d = make()
    result = plan(
        d,
        max_expansions=60,
        max_hooks=8,
        max_frontier=10,
        max_states=100,
        policy=policy,
    )
    assert result["status"] == "complete" and policy.calls > 0
    assert result["hooks"] == 2 and result["total_hooks"] == 2
    assert d.done(replay(d, result["actions"], result["events"]))
    bad = json.loads(json.dumps(result["events"]))
    bad[0]["context_after"] = {"counters": {"invented": 1}}
    with pytest.raises(InvalidAction, match="REPLAY_MISMATCH"):
        replay(d, result["actions"], bad)


def test_current_state_next_hook_keeps_original_protection():
    from fzd_shunting.policy.rule import RulePolicy

    d = make(
        [
            car(
                "guard", "修1库内", 3, target={"修1库内": {"ForceTargetPosition": [3]}}
            ),
            car("a", "A", target={"修1库内": {"ForceTargetPosition": [1, 2]}}),
        ]
    )
    d.step(d.state, d.prepare(d.state, Action("A", "get", 1)))
    snapshot = d.snapshot()
    other = Dispatcher(Environment(d.env.yard, d.env.scenario))
    other.restore(snapshot)
    predicted = next_hook(other, RulePolicy())
    assert predicted["action"] == {"line": "修1库内", "operation": "put", "count": 1}
    assert other.state.hook == 1  # prediction is not execution
    assert other.protected == {"guard"}


def test_graph_anchor_and_quota_are_visible():
    torch = pytest.importorskip("torch")
    from fzd_shunting.policy.gnn import encode_graph

    a = make(
        [
            car(
                "guard",
                "修1库内",
                2,
                target={"修1库内": {"ForceTargetPosition": [2, 3]}},
            )
        ]
    )
    b = make(
        [
            car(
                "guard",
                "修1库内",
                3,
                target={"修1库内": {"ForceTargetPosition": [2, 3]}},
            )
        ]
    )
    ga, gb = encode_graph(a, a.state, ()), encode_graph(b, b.state, ())
    assert not torch.equal(torch.from_numpy(ga.nodes), torch.from_numpy(gb.nodes))
    d = make(rules=Rules(action_limits=(ActionLimit("once", ("B",), 1),)))
    state = replace(d.state, hook=1)
    g0 = torch.from_numpy(encode_graph(d, state, (), BusinessContext()).nodes)
    g1 = torch.from_numpy(
        encode_graph(d, state, (), BusinessContext((("once", 1),))).nodes
    )
    assert not torch.equal(g0, g1)


def test_group_objective_is_feasibility_then_entire_plan_hooks():
    torch = pytest.importorskip("torch")
    from fzd_shunting.learning import trajectory_score, group_advantages

    d = make()
    initial = d.state
    result = plan(d, max_expansions=20, max_hooks=10, anytime=False)
    end = d.state
    short = trajectory_score(d, end, d.context, 100)
    long = trajectory_score(d, replace(end, hook=90), d.context, 100)
    fail = trajectory_score(d, initial, BusinessContext(), 100)
    assert short > long > fail
    assert torch.equal(group_advantages([1, 1, 1]), torch.zeros(3))
    assert group_advantages([short, long, fail])[0] > 0


def test_training_checkpoint_resume_profile_and_holdout(tmp_path):
    torch = pytest.importorskip("torch")
    from fzd_shunting.learning import train, evaluate
    from fzd_shunting.policy.gnn import load_model

    d = make()
    validation = make([car("z", "C", target="A")])
    path = tmp_path / "policy.pt"
    options = dict(
        iterations=1,
        max_hooks=5,
        seed=12,
        output=path,
        group_size=2,
        update_epochs=1,
        workers=0,
        device="cpu",
        precision="float32",
        groups_per_batch=1,
        validation=[validation],
    )
    result = train([d], **options)
    assert result["updates"] == 1
    metrics = json.loads(path.with_suffix(".metrics.json").read_text())
    assert metrics["validation_final"]["requests"] == 1
    model = load_model(path)
    model.validate_profile(d)
    with pytest.raises(ValueError, match="rules"):
        model.validate_profile(
            make(rules=Rules(action_limits=(ActionLimit("once", ("B",), 1),)))
        )
    resumed = train([d], resume=path, **options)
    assert resumed["iterations"] == 2 and resumed["updates"] == 2
    assert (
        torch.load(path, weights_only=True)["objective"]
        == "feasibility_then_total_hooks"
    )
