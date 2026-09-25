"""Lossless raw copies are kept separately; this module normalizes operating lines."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

from .domain import Scenario, State, Target, Vehicle, Yard, millimetres

MERGED = {
    "存5线北": ("存5线", 0),
    "存5北": ("存5线", 0),
    "存5线南": ("存5线", 1),
    "存5南": ("存5线", 1),
    "洗罐线北": ("洗罐线", 0),
    "洗北": ("洗罐线", 0),
    "洗罐站": ("洗罐线", 1),
    "洗南": ("洗罐线", 1),
}
ALIASES = {
    "洗油北": "洗罐油漆北",
    "机走北1线": "机北1",
    "机走北2线": "机北2",
    "机走线南": "机南",
    "机北3": "机走北",
    "机走北3线": "机走北",
    "存4线南": "存4南",
    "存4线北": "存4线",
    "存4北": "存4线",
    "机库": "机库线",
    "联6线": "联6",
    "联7线": "联7",
    "调棚": "调梁棚",
    "调棚外": "调梁线北",
    "机棚": "机走棚",
    "机棚外": "机走北",
    "预修": "预修线",
    "洗": "洗罐线",
    "油": "油漆线",
    "抛": "抛丸线",
    "轮": "卸轮线",
    "调车机": "__loco__",
    "机车": "__loco__",
    **{"存%d" % i: "存%d线" % i for i in range(1, 6)},
    **{"修%d" % i: "修%d库内" % i for i in range(1, 5)},
    **{"修%d外" % i: "修%d库外" % i for i in range(1, 5)},
}


def require_fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            label + " contains unsupported fields: " + ", ".join(sorted(unknown))
        )


def canonical(line):
    return MERGED.get(line, (ALIASES.get(line, line), 0))[0]


def target_mapping(raw):
    if not isinstance(raw, dict):
        raise ValueError(
            "TargetLines must be a mapping: {line: {ForceTargetPosition: [...]}}"
        )
    result = {}
    for line, options in raw.items():
        line = canonical(line)
        if not isinstance(options, dict) or set(options) - {"ForceTargetPosition"}:
            raise ValueError("target configuration only supports ForceTargetPosition")
        positions = options.get("ForceTargetPosition", [])
        if not isinstance(positions, list) or any(
            type(p) is not int or p < 1 for p in positions
        ):
            raise ValueError("ForceTargetPosition must contain positive integers")
        value = {"ForceTargetPosition": sorted(set(positions))} if positions else {}
        if line in result:
            # Duplicate allowed destinations are alternatives: an unrestricted one wins.
            old = result[line].get("ForceTargetPosition", [])
            value = (
                {"ForceTargetPosition": sorted(set(old + positions))}
                if old and positions
                else {}
            )
        result[line] = value
    return result


def vehicle_targets(row):
    """The train_cal3 contract has exactly one place for target positions."""
    if "ForceTargetPosition" in row:
        raise ValueError(
            "ForceTargetPosition belongs inside TargetLines[line], not on the vehicle"
        )
    return target_mapping(row["TargetLines"])


def normalize_request(raw):
    require_fields(
        raw,
        {"StartStatus", "TerminalLines", "locoNode", "SyntheticConstraints"},
        "request",
    )
    data = copy.deepcopy(raw)
    if not isinstance(data["StartStatus"], list):
        raise ValueError("StartStatus must be an array")
    groups = defaultdict(list)
    merged_names = set()
    for vehicle in data["StartStatus"]:
        # Decode targets first so misplaced positions receive the specific contract error.
        targets = vehicle_targets(vehicle)
        require_fields(
            vehicle,
            {
                "Line",
                "Position",
                "RepairProcess",
                "Type",
                "No",
                "Length",
                "IsHeavy",
                "IsWeigh",
                "IsClosedDoor",
                "TargetLines",
                "SourceLocation",
            },
            "vehicle",
        )
        for flag in ("IsHeavy", "IsWeigh", "IsClosedDoor"):
            if flag in vehicle and type(vehicle[flag]) is not bool:
                raise ValueError(flag + " must be a boolean")
        source = vehicle["Line"]
        line = canonical(source)
        position = vehicle["Position"]
        if type(position) is not int or position < 1:
            raise ValueError("Position must be a positive integer")
        if source in MERGED:
            merged_names.add(line)
            vehicle.setdefault("SourceLocation", {"Line": source, "Position": position})
        vehicle["No"] = str(vehicle["No"])
        vehicle["Line"] = line
        vehicle["TargetLines"] = targets
        groups[line].append(
            (MERGED.get(source, (line, 0))[1], position, source, vehicle)
        )
    normalized = []
    for line in sorted(groups):
        rows = sorted(groups[line], key=lambda x: (x[0], x[1], x[3]["No"]))
        if line in merged_names and any(source == line for _, _, source, _ in rows):
            raise ValueError("mixed merged and split initial locations: " + line)
        seen = set()
        for index, (part, position, source, vehicle) in enumerate(rows, 1):
            key = (part, position) if line in merged_names else position
            if key in seen:
                raise ValueError("duplicate initial position: " + line)
            seen.add(key)
            if line in merged_names:
                vehicle["Position"] = index
            normalized.append(vehicle)
    data["StartStatus"] = normalized
    loco = data.setdefault("locoNode", {"Line": "机库线", "End": "North"})
    require_fields(loco, {"Line", "End", "Vehicles"}, "locoNode")
    loco["Line"] = canonical(loco["Line"])
    if "Vehicles" in loco and not isinstance(loco["Vehicles"], list):
        raise ValueError("locoNode.Vehicles must be an array")
    terminal_names = set()
    for item in data.get("TerminalLines", []):
        require_fields(item, {"Line", "IsInspectionMode"}, "TerminalLines entry")
        if "IsInspectionMode" in item and type(item["IsInspectionMode"]) is not bool:
            raise ValueError("IsInspectionMode must be a boolean")
        item["Line"] = canonical(item["Line"])
        if item["Line"] in terminal_names:
            raise ValueError("duplicate TerminalLines entry: " + item["Line"])
        terminal_names.add(item["Line"])
    return data


def load_request(source, yard: Yard):
    raw = (
        json.loads(Path(source).read_text())
        if isinstance(source, (str, Path))
        else source
    )
    data = normalize_request(raw)
    names = yard.by_name
    if any(
        item["Line"] not in names or not names[item["Line"]].inner
        for item in data.get("TerminalLines", [])
    ):
        raise ValueError(
            "TerminalLines inspection mode applies only to known inner depots"
        )
    vehicles = []
    stacks = {name: [] for name in names}
    explicit_train = tuple(str(x) for x in data["locoNode"].get("Vehicles", []))
    train_rows = []
    seen = set()
    for row in data["StartStatus"]:
        no, line = row["No"], row["Line"]
        if no in seen:
            raise ValueError("duplicate vehicle: " + no)
        seen.add(no)
        targets = tuple(
            Target(k, tuple(v.get("ForceTargetPosition", [])))
            for k, v in row["TargetLines"].items()
        )
        if not targets or any(
            t.line not in names or not names[t.line].parking for t in targets
        ):
            raise ValueError("missing or unknown parking destination: " + no)
        if line not in names and line != "__loco__":
            raise ValueError("unknown source line: " + line)
        if line != "__loco__" and not names[line].parking:
            raise ValueError("vehicles cannot be parked on a transit line")
        car = Vehicle(
            no,
            millimetres(row["Length"]),
            line,
            row["Position"],
            targets,
            bool(row.get("IsHeavy", False)),
            bool(row.get("IsClosedDoor", False)),
            bool(row.get("IsWeigh", False)),
            row.get("RepairProcess", ""),
            row.get("Type", ""),
        )
        vehicles.append(car)
        if line == "__loco__":
            train_rows.append((row["Position"], no))
        else:
            stacks[line].append((row["Position"], no))
    derived_train = tuple(no for _, no in sorted(train_rows))
    if "Vehicles" in data["locoNode"] and explicit_train != derived_train:
        raise ValueError(
            "locoNode.Vehicles must agree with StartStatus Line=机车 and Position order"
        )
    train = explicit_train or derived_train
    loco = data["locoNode"]
    if loco["Line"] not in names or loco.get("End", "North") not in ("North", "South"):
        raise ValueError("invalid locomotive start")
    state = State(
        tuple(
            (name, tuple(no for _, no in sorted(rows)))
            for name, rows in sorted(stacks.items())
        ),
        train,
        loco["Line"],
        loco.get("End", "North"),
    )
    modes = {
        v["Line"]: bool(v.get("IsInspectionMode", False))
        for v in data.get("TerminalLines", [])
    }
    capacities = tuple(
        (t.name, 7 if modes.get(t.name, False) else 5) for t in yard.tracks if t.inner
    )
    return Scenario(
        tuple(vehicles),
        state,
        capacities,
        Path(source).stem if isinstance(source, (str, Path)) else "inline",
    )
