import copy
import json
from dataclasses import replace

import pytest

from fzd_shunting.data import augment_request, prepare_data
from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.domain import Action, Gate, InvalidAction, Rules, Track, Yard
from fzd_shunting.environment import Environment
from fzd_shunting.planning import plan, replay
from fzd_shunting.request import load_request, normalize_request


def car(no, line, position, targets=None, length=13.2, **kwargs):
    return dict(
        No=no,
        Line=line,
        Position=position,
        Length=length,
        TargetLines=(
            {x: {} for x in (targets or ["B"])}
            if isinstance(targets, (list, type(None)))
            else targets
        ),
        **kwargs,
    )


@pytest.fixture
def yard():
    tracks = [
        Track("R", 0, 0, "transit"),
        Track("A", 300000, 20),
        Track("B", 300000, 20),
        Track("C", 300000, 20),
        Track("修1库外", 118400, 9),
        Track("修1库内", 151700, 7),
    ]
    edges = (
        ("A", "R"),
        ("B", "R"),
        ("C", "R"),
        ("修1库外", "R"),
        ("修1库内", "修1库外"),
    )
    return Yard(tuple(tracks), edges)


def dispatcher(yard, rows, rules=Rules(), start="R", end="North"):
    data = {"StartStatus": rows, "locoNode": {"Line": start, "End": end}}
    return Dispatcher(Environment(yard, load_request(data, yard)), rules)


def step(d, state, line, operation, count):
    return d.step(state, d.prepare(state, Action(line, operation, count)))


def test_merged_north_then_south_idempotent():
    rows = [
        car("s2", "存5线南", 7, ["存5线南"]),
        car("n2", "存5线北", 5),
        car("s1", "存5线南", 1),
        car("n1", "存5线北", 2),
        car("ws", "洗罐站", 1),
        car("wn", "洗罐线北", 3),
        car("anchor", "修1库内", 4, ["修1库内"]),
    ]
    result = normalize_request({"StartStatus": rows})
    merged = [v for v in result["StartStatus"] if v["Line"] == "存5线"]
    assert [v["No"] for v in merged] == ["n1", "n2", "s1", "s2"]
    assert [v["Position"] for v in merged] == [1, 2, 3, 4]
    assert [v["No"] for v in result["StartStatus"] if v["Line"] == "洗罐线"] == [
        "wn",
        "ws",
    ]
    assert (
        next(v for v in result["StartStatus"] if v["No"] == "anchor")["Position"] == 4
    )
    assert result == normalize_request(result)
    assert rows[0]["Line"] == "存5线南"


def test_get_can_span_old_split_boundary_in_one_hook():
    yard = Yard.load()
    data = {
        "StartStatus": [
            car("south", "存5线南", 1, ["存2线"]),
            car("north", "存5线北", 1, ["存2线"]),
        ],
        "locoNode": {"Line": "存5线北", "End": "North"},
    }
    d = Dispatcher(Environment(yard, load_request(data, yard)))
    state = step(d, d.env.scenario.initial, "存5线", "get", 2)
    assert state.train == ("north", "south") and state.hook == 1


def test_get_put_blocks_and_preloaded_loco(yard):
    d = dispatcher(
        yard,
        [
            car("a", "A", 1),
            car("b", "A", 2),
            car("c", "A", 3),
            car("d", "机车", 1),
            car("e", "机车", 2),
        ],
    )
    state = step(d, d.env.scenario.initial, "A", "get", 2)
    assert state.stack("A") == ("c",)
    assert state.train == ("d", "e", "a", "b")
    state = step(d, state, "B", "put", 2)
    assert state.stack("B") == ("a", "b")
    assert state.train == ("d", "e")
    assert state.hook == 2
    assert len(d.events) == 2


def test_length_limit_includes_existing_train_and_loco(yard):
    d = dispatcher(yard, [car("a", "A", 1, length=88), car("b", "机车", 1, length=90)])
    initial = d.env.scenario.initial
    assert d.prepare(initial, Action("A", "get", 1)).after.train == ("b", "a")
    d2 = dispatcher(
        yard, [car("a", "A", 1, length=88.001), car("b", "机车", 1, length=90)]
    )
    with pytest.raises(InvalidAction, match="TRAIN_LENGTH"):
        d2.prepare(d2.env.scenario.initial, Action("A", "get", 1))
    assert initial == d.env.scenario.initial and not d.events


def test_heavy_is_normal_and_twenty_limit(yard):
    d = dispatcher(
        yard, [car(str(i), "A", i + 1, length=5, IsHeavy=True) for i in range(21)]
    )
    assert (
        len(d.prepare(d.env.scenario.initial, Action("A", "get", 20)).after.train) == 20
    )
    with pytest.raises(InvalidAction, match="TRAIN_COUNT"):
        d.prepare(d.env.scenario.initial, Action("A", "get", 21))


def test_north_blocking_and_inner_outer_two_hooks(yard):
    d = dispatcher(yard, [car("outer", "修1库外", 1), car("inner", "修1库内", 1)])
    state = d.env.scenario.initial
    with pytest.raises(InvalidAction, match="NORTH_ROUTE_BLOCKED"):
        d.prepare(state, Action("修1库内", "get", 1))
    state = step(d, state, "修1库外", "get", 1)
    assert state.stack("修1库内") == ("inner",)
    state = step(d, state, "修1库内", "get", 1)
    assert state.train == ("outer", "inner") and state.hook == 2


def test_put_inner_and_outer_is_two_hooks(yard):
    d = dispatcher(
        yard, [car("o", "机车", 1, ["修1库外"]), car("i", "机车", 2, ["修1库内"])]
    )
    state = step(d, d.env.scenario.initial, "修1库内", "put", 1)
    state = step(d, state, "修1库外", "put", 1)
    assert state.stack("修1库内") == ("i",)
    assert state.stack("修1库外") == ("o",)
    assert state.hook == 2


def test_south_start_must_reach_north_port(yard):
    d = dispatcher(yard, [car("a", "A", 1)], start="A", end="South")
    with pytest.raises(InvalidAction, match="NORTH_ROUTE_BLOCKED"):
        d.prepare(d.env.scenario.initial, Action("A", "get", 1))
    north = replace(d.env.scenario.initial, loco_end="North")
    assert d.prepare(north, Action("A", "get", 1))


def test_protection_initial_position_and_whole_inner_cache_ban(yard):
    d = dispatcher(
        yard,
        [
            car("guard", "修1库内", 3, {"修1库内": {"ForceTargetPosition": [3]}}),
            car("buffer", "机车", 1, ["B"]),
        ],
    )
    assert d.protected == {"guard"}
    state = d.env.scenario.initial
    with pytest.raises(InvalidAction, match="PROTECTED"):
        d.prepare(state, Action("修1库内", "get", 1))
    with pytest.raises(InvalidAction, match="NO_BUFFER"):
        d.prepare(state, Action("修1库内", "put", 1))


def test_final_put_before_protected_vehicle_is_allowed(yard):
    d = dispatcher(
        yard,
        [
            car("guard", "修1库内", 3, {"修1库内": {"ForceTargetPosition": [3]}}),
            car("arrive", "机车", 1, {"修1库内": {"ForceTargetPosition": [1, 2]}}),
        ],
    )
    state = step(d, d.env.scenario.initial, "修1库内", "put", 1)
    assert d.done(state)
    assert d.order_assignment(state.stack("修1库内"), "修1库内") == (
        ("arrive", 1),
        ("guard", 3),
    )


def test_unprotected_inner_cache_151_7_and_final_slot_count(yard):
    d = dispatcher(yard, [car(str(i), "机车", i + 1, length=20) for i in range(7)])
    candidate = d.prepare(d.env.scenario.initial, Action("修1库内", "put", 7))
    assert len(candidate.after.stack("修1库内")) == 7
    assert not d.done(candidate.after)
    d2 = dispatcher(yard, [car(str(i), "机车", i + 1, length=19) for i in range(8)])
    with pytest.raises(InvalidAction, match="TRACK_LENGTH"):
        d2.prepare(d2.env.scenario.initial, Action("修1库内", "put", 8))


def test_relative_slots_allow_empty_positions_and_alternatives(yard):
    d = dispatcher(
        yard,
        [
            car("a", "B", 1, {"B": {"ForceTargetPosition": [2, 6]}}),
            car("b", "B", 2, {"B": {"ForceTargetPosition": [5]}}),
            car("c", "B", 3, {"B": {"ForceTargetPosition": [8, 9]}}),
        ],
    )
    assert d.done(d.env.scenario.initial)
    assert d.order_assignment(("a", "b", "c"), "B") == (("a", 2), ("b", 5), ("c", 8))
    assert d.order_assignment(("b", "a", "c"), "B") == (("b", 5), ("a", 6), ("c", 8))
    assert d.order_assignment(("c", "a", "b"), "B") is None


def test_gate_inclusive_does_not_force_operation(yard):
    rules = Rules((Gate(("B",), 3),))
    d = dispatcher(yard, [car("a", "A", 1)], rules)
    state = step(d, d.env.scenario.initial, "A", "get", 1)
    with pytest.raises(InvalidAction, match="HOOK_GATE"):
        d.prepare(state, Action("B", "put", 1))
    state = step(d, state, "C", "put", 1)
    assert any(
        c.action.line == "B"
        for c in d.candidates(
            replace(
                state,
                train=("a",),
                stacks=tuple((line, ()) for line, _ in state.stacks),
            )
        )
    )
    assert d.prepare(state, Action("C", "get", 1))


def test_stale_and_forged_candidate(yard):
    d = dispatcher(yard, [car("a", "A", 1)])
    initial = d.env.scenario.initial
    candidate = d.prepare(initial, Action("A", "get", 1))
    forged = replace(candidate, after=initial)
    after = d.step(initial, forged)
    assert after.hook == 1 and after.train == ("a",)
    with pytest.raises(InvalidAction, match="STALE"):
        d.step(after, candidate)


def test_input_capacity_is_reported_not_relaxed(yard):
    d = dispatcher(yard, [car("a", "修1库外", 1, length=118.401)])
    assert "TRACK_LENGTH" in d.env.validate_state(d.env.scenario.initial)[0]
    assert plan(d)["status"] == "invalid_request"


def test_complete_plan_replays_and_tampered_log_fails(yard):
    d = dispatcher(yard, [car("a", "A", 1, ["B"]), car("b", "A", 2, ["C"])])
    result = plan(d, max_expansions=50)
    assert result["status"] == "complete"
    assert len(result["actions"]) == 3
    stored = json.loads(json.dumps(result))
    assert d.done(replay(d, stored["actions"], stored["events"]))
    stored["events"][0]["route"] = ["invented"]
    with pytest.raises(InvalidAction, match="REPLAY_MISMATCH"):
        replay(d, stored["actions"], stored["events"])


def test_augmentation_deterministic_and_matching(yard):
    raw = {
        "StartStatus": [car("a", "A", 1, ["B", "C"]), car("b", "A", 2, ["B"])],
        "locoNode": {"Line": "R", "End": "North"},
    }
    normalized = normalize_request(raw)
    a, metadata = augment_request(normalized, yard, 123, 1)
    assert a == augment_request(normalized, yard, 123, 1)[0]
    assert metadata["slot_matching_found"] and metadata["constraints_added"] == 3
    assert normalized == normalize_request(raw)


def test_raw_copy_is_byte_identical(tmp_path, yard):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    text = json.dumps({"StartStatus": [car("a", "A", 1)], "locoNode": {"Line": "R"}})
    (source / "sample.json").write_text(text)
    result = prepare_data(source, dest, yard, fraction=1)
    assert (dest / "raw" / "sample.json").read_bytes() == (
        source / "sample.json"
    ).read_bytes()
    assert (dest / "normalized" / "sample.json").exists()
    assert result["files"][0]["constraints_added"] == 1


def test_terminal_eligibility_is_trusted_from_request(yard):
    temporary = replace(
        yard,
        tracks=tuple(
            replace(t, kind="temporary") if t.name == "B" else t for t in yard.tracks
        ),
    )
    d = dispatcher(temporary, [car("a", "B", 1, ["B"])])
    assert d.done(d.env.scenario.initial)


def test_temporary_and_terminal_lengths_are_separate(yard):
    modified = replace(
        yard,
        tracks=tuple(
            replace(t, terminal_length_mm=20000) if t.name == "B" else t
            for t in yard.tracks
        ),
    )
    d = dispatcher(modified, [car("a", "机车", 1), car("b", "机车", 2)])
    state = step(d, d.env.scenario.initial, "B", "put", 2)
    assert state.stack("B") == ("a", "b")
    assert not d.done(state)


def test_transit_cannot_be_put_target(yard):
    d = dispatcher(yard, [car("a", "机车", 1)])
    with pytest.raises(InvalidAction, match="ACTION_TRACK"):
        d.prepare(d.env.scenario.initial, Action("R", "put", 1))
