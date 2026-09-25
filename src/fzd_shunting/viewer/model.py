"""Read-only viewer for the single train_cal3 request/plan contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..domain import State
from ..request import normalize_request


def _parts(data):
    if not isinstance(data, dict):
        raise ValueError("需要 train_cal3 请求、计划或 {request, plan} 对象")
    if set(data) == {"request", "plan"}:
        request, plan = data["request"], data["plan"]
        if not isinstance(request, dict) or "StartStatus" not in request:
            raise ValueError("request 必须包含 StartStatus")
    elif "StartStatus" in data:
        request, plan = data, None
    elif "actions" in data or "events" in data:
        request, plan = None, data
    else:
        raise ValueError(
            "未识别格式；只支持 train_cal3 的 StartStatus、actions/events 和 {request, plan}"
        )
    if plan is not None:
        if not isinstance(plan, dict) or plan.get("schema_version") != 2:
            raise ValueError("计划必须使用 train_cal3 schema_version=2")
        if set(plan) & {"Operations", "Data", "Response"}:
            raise ValueError("不支持旧格式计划")
    return request, plan


def _request_state(raw):
    data = normalize_request(raw)
    vehicles = {}
    stacks = {}
    train = []
    for row in data["StartStatus"]:
        no = row["No"]
        line = row["Line"]
        targets = row["TargetLines"]
        if not no or no in vehicles:
            raise ValueError("请求有空车号或重复车号：" + no)
        protected = line in {"修%d库内" % n for n in range(1, 5)} and row[
            "Position"
        ] in targets.get(line, {}).get("ForceTargetPosition", [])
        length = row.get("Length")
        if type(length) not in (int, float) or length <= 0:
            raise ValueError("车辆 Length 必须为正数")
        vehicles[no] = {
            "no": no,
            "length": length,
            "type": row.get("Type"),
            "repair": row.get("RepairProcess"),
            "heavy": row.get("IsHeavy"),
            "weigh": row.get("IsWeigh"),
            "closed": row.get("IsClosedDoor"),
            "targets": targets,
            "protected": protected,
            "initial_line": line,
            "initial_position": row["Position"],
            "raw": copy.deepcopy(row),
        }
        if line == "__loco__":
            train.append(no)
        else:
            stacks.setdefault(line, []).append(no)
    loco = data["locoNode"]
    if "Vehicles" in loco and [str(x) for x in loco["Vehicles"]] != train:
        raise ValueError("locoNode.Vehicles 与 StartStatus 机车车列不一致")
    return {
        "stacks": stacks,
        "train": train,
        "loco_line": loco["Line"],
        "loco_end": loco.get("End", "North"),
    }, vehicles


def _cars(state):
    return state["train"] + [c for cars in state["stacks"].values() for c in cars]


def _snapshot(raw):
    state = State.from_dict(raw)
    result = {
        "stacks": {line: list(cars) for line, cars in state.stacks},
        "train": list(state.train),
        "loco_line": state.loco_line,
        "loco_end": state.loco_end,
    }
    if len(state.stacks) != len(result["stacks"]) or len(_cars(result)) != len(
        set(_cars(result))
    ):
        raise ValueError("快照有重复股道或车辆")
    return result


def _same(a, b):
    return (
        a["train"] == b["train"]
        and a["loco_line"] == b["loco_line"]
        and a["loco_end"] == b["loco_end"]
        and {k: v for k, v in a["stacks"].items() if v}
        == {k: v for k, v in b["stacks"].items() if v}
    )


def _operation(raw, index):
    action = raw.get("action", raw)
    if not isinstance(action, dict) or set(action) != {"line", "operation", "count"}:
        raise ValueError("动作仅包含 line、operation、count")
    if (
        action["operation"] not in ("get", "put")
        or type(action["count"]) is not int
        or action["count"] < 1
    ):
        raise ValueError("动作须为 get/put，数量为正整数")
    if "action" in raw and raw.get("schema_version") != 2:
        raise ValueError("事件必须使用 schema_version=2")
    for field in ("vehicles", "route"):
        if field in raw and not isinstance(raw[field], (list, tuple)):
            raise ValueError(field + " 必须为数组")
    return {
        **action,
        "vehicles": list(raw.get("vehicles", [])),
        "route": list(raw.get("route", [])),
        "index": raw.get("hook", index),
    }


def _transfer(before, op):
    after = copy.deepcopy(before)
    stack = after["stacks"].setdefault(op["line"], [])
    count = op["count"]
    source = stack if op["operation"] == "get" else after["train"]
    if len(source) < count:
        raise ValueError("来源车辆不足，无法重建本步")
    moved = source[:count] if op["operation"] == "get" else source[-count:]
    if op["vehicles"] and moved != op["vehicles"]:
        raise ValueError("vehicles 与北端前缀/机车尾部后缀不一致")
    if op["operation"] == "get":
        del stack[:count]
        after["train"].extend(moved)
    else:
        del after["train"][-count:]
        stack[:0] = moved
    after["loco_line"] = op["line"]
    after["loco_end"] = "North"
    return after, moved


def build_view(primary, companion=None, title="调车查看器"):
    request, plan = _parts(primary)
    if companion is not None:
        r, p = _parts(companion)
        if request is not None and r is not None and request != r:
            raise ValueError("两份不同的请求不能配对")
        if plan is not None and p is not None and plan != p:
            raise ValueError("两份不同的计划不能配对")
        request = request if request is not None else r
        plan = plan if plan is not None else p
    issues = []
    vehicles = {}
    state = None
    if request is not None:
        state, vehicles = _request_state(request)
    operations = (plan or {}).get("events", (plan or {}).get("actions", []))
    if plan is not None and "events" in plan and "actions" in plan:
        event_actions = [event.get("action") for event in plan["events"]]
        if event_actions != plan["actions"]:
            raise ValueError("events 与 actions 的动作序列不一致")
    if not isinstance(operations, list) or any(
        not isinstance(o, dict) for o in operations
    ):
        raise ValueError("计划动作须为对象数组")
    initial_snapshot = (plan or {}).get("initial_snapshot", {}).get("state")
    recorded = operations[0].get("before") if operations else initial_snapshot
    if recorded is None:
        recorded = initial_snapshot
    if recorded is not None:
        snapshot = _snapshot(recorded)
        if state is not None and not _same(state, snapshot):
            issues.append(
                "请求初态与回放起点不同，采用计划快照；可能是从中途状态继续的计划"
            )
        state = snapshot
    yard = json.loads((Path(__file__).parents[1] / "yard.json").read_text())
    layout = json.loads((Path(__file__).parent / "assets/layout.json").read_text())
    tracks = {
        t["name"]: {**t, **layout["tracks"].get(t["name"], {})} for t in yard["tracks"]
    }
    result = {
        "title": title,
        "vehicles": vehicles,
        "tracks": tracks,
        "frames": [],
        "issues": issues,
        "needs_request": state is None,
        "has_plan": plan is not None,
        "total_steps": len(operations),
        "total_hooks": len(operations),
        "status": (plan or {}).get("status"),
        "stopped": None,
        "canvas": layout["canvas"],
        "replay_note": (
            "展示 train_cal3 状态与取放记录。"
            if plan is not None
            else "初始请求 · 车序由北向南。"
        ),
    }
    if state is None:
        issues.append("此计划缺少初态，请搭配对应的规范请求")
        return result
    initial_hook = (recorded or {}).get("hook", 0)
    result["frames"].append(
        {
            **copy.deepcopy(state),
            "hook": initial_hook,
            "action": None,
            "issues": [],
            "source": "initial",
        }
    )
    for i, raw in enumerate(operations, 1):
        notes = []
        try:
            op = _operation(raw, i)
            if "after" in raw:
                after = _snapshot(raw["after"])
                if "before" not in raw:
                    raise ValueError("事件缺少 before 快照")
                if not _same(state, _snapshot(raw["before"])):
                    notes.append("本步 before 与上一状态不连续")
                try:
                    derived, moved = _transfer(state, op)
                    if not _same(derived, after):
                        notes.append("动作推演与 after 不符，按原始快照展示")
                except ValueError as exc:
                    notes.append(str(exc) + "；按原始快照展示")
                    moved = op["vehicles"]
                if set(_cars(after)) != set(_cars(state)):
                    notes.append("快照中的车辆集合发生变化")
                source = "snapshot"
            else:
                after, moved = _transfer(state, op)
                source = "derived"
            op["vehicles"] = moved
            if any(vehicles.get(c, {}).get("protected") for c in moved):
                notes.append("本步移动了保护车")
            result["frames"].append(
                {
                    **after,
                    "hook": raw.get("hook", initial_hook + i),
                    "action": op,
                    "issues": notes,
                    "source": source,
                }
            )
            state = after
        except (ValueError, KeyError, TypeError) as exc:
            result["stopped"] = f"第 {i} 步无法重建：{exc}。已保留前 {i-1} 步。"
            issues.append(result["stopped"])
            break
    for frame in result["frames"]:
        for no in _cars(frame):
            vehicles.setdefault(
                no, {"no": no, "length": None, "targets": {}, "raw": {}}
            )
        for name in [*frame["stacks"], frame["loco_line"]]:
            tracks.setdefault(name, {"name": name, "kind": "unknown"})
    for v in vehicles.values():
        for name in v["targets"]:
            tracks.setdefault(name, {"name": name, "kind": "unknown"})
        if v.get("protected"):
            tracks[v["initial_line"]]["protected_no_buffer"] = True
    return result
