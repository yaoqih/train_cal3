import copy
import json
from pathlib import Path

import pytest

from fzd_shunting.viewer import build_view, render_html

ROOT = Path(__file__).resolve().parents[1]


def request(*cars):
    return {"StartStatus": list(cars), "locoNode": {"Line": "机库线"}}


def car(no, line="存1线", pos=1, **kw):
    return {
        "No": no,
        "Line": line,
        "Position": pos,
        "Length": 13.2,
        "TargetLines": {"存2线": {}},
        **kw,
    }


def plan(actions=None, events=None):
    value = {"schema_version": 2}
    if actions is not None:
        value["actions"] = actions
    if events is not None:
        value["events"] = events
    return value


def test_get_put_order_and_conservation_with_preloaded_locomotive():
    source = request(car("A"), car("B", pos=2), car("C", pos=3), car("D", "机车"))
    actions = [
        {"line": "存1线", "operation": "get", "count": 2},
        {"line": "存2线", "operation": "put", "count": 2},
    ]
    untouched = copy.deepcopy(source)
    view = build_view(plan(actions), source)
    assert source == untouched
    assert view["frames"][1]["train"] == ["D", "A", "B"]
    assert view["frames"][-1]["train"] == ["D"]
    assert view["frames"][-1]["stacks"]["存2线"] == ["A", "B"]
    for f in view["frames"]:
        assert sorted(
            f["train"] + [c for ids in f["stacks"].values() for c in ids]
        ) == ["A", "B", "C", "D"]


def test_request_merge_and_protection():
    view = build_view(
        request(
            car("S", "洗罐站"),
            car("N", "洗罐线北", 5),
            car(
                "P", "修1库内", 2, TargetLines={"修1库内": {"ForceTargetPosition": [2]}}
            ),
            car("O", "修1库外"),
        )
    )
    assert view["frames"][0]["stacks"]["洗罐线"] == ["N", "S"]
    assert view["vehicles"]["P"]["protected"]
    assert view["tracks"]["修1库内"]["protected_no_buffer"]
    assert view["frames"][0]["stacks"]["修1库外"] == ["O"]


def test_actions_need_request_snapshots_do_not():
    action = {"line": "存1线", "operation": "get", "count": 1}
    assert build_view(plan([action]))["needs_request"]
    before = {"stacks": {"存1线": ["A"]}, "train": [], "loco_line": "机库线"}
    after = {"stacks": {"存1线": []}, "train": ["A"], "loco_line": "存1线"}
    view = build_view(
        plan(
            events=[
                {
                    "schema_version": 2,
                    "action": action,
                    "before": before,
                    "after": after,
                }
            ]
        )
    )
    assert len(view["frames"]) == 2
    assert view["vehicles"]["A"]["length"] is None
    assert view["frames"][1]["action"]["vehicles"] == ["A"]


def test_conflicting_actions_and_events_rejected():
    with pytest.raises(ValueError, match="不一致"):
        build_view(
            plan(
                actions=[{"line": "存1线", "operation": "get", "count": 1}], events=[]
            ),
            request(car("A")),
        )


def test_broken_plan_retains_prefix():
    view = build_view(
        plan(
            [
                {"line": "存1线", "operation": "get", "count": 1},
                {"line": "存1线", "operation": "get", "count": 9},
            ]
        ),
        request(car("A")),
    )
    assert len(view["frames"]) == 2 and view["stopped"].startswith("第 2 步")


def test_authoritative_snapshot_flags_discontinuous_inventory():
    view = build_view(
        plan(
            events=[
                {
                    "schema_version": 2,
                    "action": {"line": "存1线", "operation": "get", "count": 1},
                    "before": {
                        "stacks": {"存1线": ["A"]},
                        "train": [],
                        "loco_line": "机库线",
                    },
                    "after": {"stacks": {}, "train": ["Z"], "loco_line": "存1线"},
                }
            ]
        )
    )
    assert view["frames"][1]["train"] == ["Z"]
    assert any("车辆集合" in s for s in view["frames"][1]["issues"])


def test_overcapacity_is_visible_without_certifying_legality():
    view = build_view(request(car("A", Length=200), car("B", "外部股道")))
    assert (
        len(view["vehicles"]) == 2 and view["tracks"]["外部股道"]["kind"] == "unknown"
    )


@pytest.mark.parametrize(
    "data",
    [
        {"Request": request(car("A")), "Response": {"Operations": []}},
        {"Operations": []},
        {"actions": []},
        request(car("A", ForceTargetPosition=[2])),
        request(car("A", TargetLines=["存2线"])),
    ],
)
def test_no_legacy_fallback(data):
    with pytest.raises(ValueError):
        build_view(data)


def test_xss_safe_self_contained_export():
    attack = '</script><script>alert("xss")</script>'
    html = render_html(build_view(request(car(attack)), title=attack))
    assert attack not in html
    payload = html.split('<script id="viewer-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(payload)["title"] == attack
    assert 'src="http' not in html and "/*__" not in html


def test_all_normalized_requests_are_viewable():
    files = list((ROOT / "data/point_to_area/normalized").glob("*.json"))
    assert files
    for path in files:
        data = json.loads(path.read_text())
        view = build_view(data)
        assert len(view["vehicles"]) == len(data["StartStatus"])


def test_demo_is_portable():
    view = build_view(json.loads((ROOT / "scenarios/viewer-demo.json").read_text()))
    assert len(view["frames"]) == 7 and not view["issues"]


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="重复车号"):
        build_view(request(car("A"), car("A", pos=2)))
