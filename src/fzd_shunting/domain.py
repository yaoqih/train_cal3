from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def millimetres(value) -> int:
    mm = Decimal(str(value)) * 1000
    if not mm.is_finite() or mm <= 0 or mm != mm.to_integral_value():
        raise ValueError("length must be positive and expressible in millimetres")
    return int(mm)


@dataclass(frozen=True)
class Track:
    name: str
    length_mm: int
    reference_slots: int = 20
    kind: str = "storage"
    terminal_length_mm: Optional[int] = None

    @property
    def inner(self):
        return self.name in {"修%d库内" % i for i in range(1, 5)}

    @property
    def parking(self):
        return self.kind != "transit"

    @property
    def final_capacity_mm(self):
        return (
            self.length_mm
            if self.terminal_length_mm is None
            else self.terminal_length_mm
        )


@dataclass(frozen=True)
class Yard:
    tracks: tuple[Track, ...]
    edges: tuple[tuple[str, str], ...]  # south node, north node
    locomotive_mm: int = 15000
    train_limit_mm: int = 193000
    train_count_limit: int = 20
    route_model: str = "north-end-occupancy-v1"

    @property
    def by_name(self):
        return {t.name: t for t in self.tracks}

    @classmethod
    def load(cls, path: Optional[str] = None):
        data = json.loads(
            Path(path or Path(__file__).with_name("yard.json")).read_text()
        )
        unknown = set(data) - {
            "schema_version",
            "notes",
            "tracks",
            "edges",
            "locomotive_mm",
            "train_limit_mm",
            "train_count_limit",
            "route_model",
        }
        if unknown:
            raise ValueError("unknown yard fields: " + ", ".join(sorted(unknown)))
        if data.get("schema_version", 1) != 1:
            raise ValueError("unsupported yard schema")
        for track in data["tracks"]:
            if set(track) - {
                "name",
                "length_mm",
                "reference_slots",
                "kind",
                "terminal_length_mm",
            }:
                raise ValueError("unknown track fields")
            if not isinstance(track["name"], str) or not track["name"]:
                raise ValueError("track name must be a nonempty string")
            if track["kind"] not in {"storage", "temporary", "operation", "transit"}:
                raise ValueError("unknown track kind")
            for field in ("length_mm", "reference_slots", "terminal_length_mm"):
                if field in track and (
                    type(track[field]) is not int or track[field] < 0
                ):
                    raise ValueError("track lengths/slots must be nonnegative integers")
        tracks = tuple(
            Track(
                x["name"],
                int(x["length_mm"]),
                x.get("reference_slots", 20),
                x["kind"],
                x.get("terminal_length_mm"),
            )
            for x in data["tracks"]
        )
        names = {t.name for t in tracks}
        if (
            len(names) != len(tracks)
            or any(
                t.length_mm < 0 or t.reference_slots < 0 or t.final_capacity_mm < 0
                for t in tracks
            )
            or any(a not in names or b not in names or a == b for a, b in data["edges"])
        ):
            raise ValueError("invalid yard nodes or edges")
        limits = {
            k: data.get(k, default)
            for k, default in (
                ("locomotive_mm", 15000),
                ("train_limit_mm", 193000),
                ("train_count_limit", 20),
            )
        }
        if any(type(v) is not int or v <= 0 for v in limits.values()):
            raise ValueError("yard limits must be positive integers")
        if limits["train_limit_mm"] <= limits["locomotive_mm"]:
            raise ValueError("train limit must exceed locomotive length")
        return cls(
            tracks,
            tuple(tuple(e) for e in data["edges"]),
            **limits,
            route_model=data.get("route_model", "north-end-occupancy-v1"),
        )


@dataclass(frozen=True)
class Target:
    line: str
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class Vehicle:
    no: str
    length_mm: int
    initial_line: str
    initial_position: int
    targets: tuple[Target, ...]
    heavy: bool = False
    closed_door: bool = False
    weigh: bool = False
    repair_process: str = ""
    vehicle_type: str = ""

    def target(self, line):
        return next((t for t in self.targets if t.line == line), None)


@dataclass(frozen=True)
class State:
    stacks: tuple[tuple[str, tuple[str, ...]], ...]
    train: tuple[str, ...] = ()
    loco_line: str = "机库线"
    loco_end: str = "North"
    hook: int = 0

    def stack(self, line):
        return dict(self.stacks).get(line, ())

    def to_dict(self):
        return {
            "stacks": dict(self.stacks),
            "train": self.train,
            "loco_line": self.loco_line,
            "loco_end": self.loco_end,
            "hook": self.hook,
        }

    def fingerprint(self):
        data = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict) or set(raw) - {
            "stacks",
            "train",
            "loco_line",
            "loco_end",
            "hook",
        }:
            raise ValueError("invalid state fields")
        stacks = raw["stacks"]
        if not isinstance(stacks, dict):
            raise ValueError("state stacks must be a track-to-vehicles mapping")
        if any(
            not isinstance(v, (list, tuple)) for v in stacks.values()
        ) or not isinstance(raw.get("train", []), (list, tuple)):
            raise ValueError("state vehicle sequences must be arrays")
        return cls(
            tuple(
                sorted((str(k), tuple(str(c) for c in v)) for k, v in stacks.items())
            ),
            tuple(str(c) for c in raw.get("train", [])),
            raw["loco_line"],
            raw.get("loco_end", "North"),
            raw.get("hook", 0),
        )


@dataclass(frozen=True)
class Scenario:
    vehicles: tuple[Vehicle, ...]
    initial: State
    inner_capacities: tuple[tuple[str, int], ...]
    name: str = ""

    @property
    def cars(self):
        return {v.no: v for v in self.vehicles}

    @property
    def protected(self):
        inner = dict(self.inner_capacities)
        return frozenset(
            v.no
            for v in self.vehicles
            if v.initial_line in inner
            and v.target(v.initial_line) is not None
            and v.initial_position in v.target(v.initial_line).positions
        )


@dataclass(frozen=True)
class Action:
    line: str
    operation: str
    count: int

    def to_dict(self):
        return asdict(self)


class InvalidAction(ValueError):
    pass


@dataclass(frozen=True)
class Gate:
    lines: tuple[str, ...]
    earliest_hook: int
    operations: tuple[str, ...] = ("get", "put")


@dataclass(frozen=True)
class ActionLimit:
    name: str
    lines: tuple[str, ...]
    max_actions: int
    operations: tuple[str, ...] = ("put",)


@dataclass(frozen=True)
class BusinessContext:
    """Only history-dependent facts belong here; no duplicate vehicle state."""

    counters: tuple[tuple[str, int], ...] = ()

    def count(self, key):
        return dict(self.counters).get(key, 0)

    def increment(self, keys):
        counts = dict(self.counters)
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        return BusinessContext(tuple(sorted(counts.items())))

    def to_dict(self):
        return {"counters": dict(self.counters)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"counters"}:
            raise ValueError("business context must contain counters")
        counts = value.get("counters", {})
        if not isinstance(counts, dict) or any(
            type(v) is not int or v < 0 for v in counts.values()
        ):
            raise ValueError("business counters must be nonnegative integers")
        return cls(tuple(sorted(counts.items())))


@dataclass(frozen=True)
class Rules:
    gates: tuple[Gate, ...] = ()
    action_limits: tuple[ActionLimit, ...] = ()

    def earliest(self, line, operation):
        return max(
            [1]
            + [
                g.earliest_hook
                for g in self.gates
                if line in g.lines and operation in g.operations
            ]
        )

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        from .request import canonical

        data = json.loads(Path(path).read_text())
        if set(data) - {"access_rules", "action_limits"}:
            raise ValueError(
                "unsupported business rule fields: "
                + str(sorted(set(data) - {"access_rules", "action_limits"}))
            )
        gates = []
        for rule in data.get("access_rules", []):
            if set(rule) - {"name", "lines", "earliest_hook", "operations"}:
                raise ValueError("unsupported access rule fields")
            n = rule["earliest_hook"]
            if type(n) is not int or n < 1:
                raise ValueError("earliest_hook must be a positive integer")
            ops = tuple(rule.get("operations", ["get", "put"]))
            if not ops or not set(ops) <= {"get", "put"}:
                raise ValueError("invalid gate operations")
            gates.append(Gate(tuple(canonical(x) for x in rule["lines"]), n, ops))
        limits = []
        for rule in data.get("action_limits", []):
            if set(rule) - {"name", "lines", "max_actions", "operations"}:
                raise ValueError("unsupported action limit fields")
            n = rule["max_actions"]
            ops = tuple(rule.get("operations", ["put"]))
            if type(n) is not int or n < 0 or not ops or not set(ops) <= {"get", "put"}:
                raise ValueError("invalid action limit")
            limits.append(
                ActionLimit(
                    rule["name"], tuple(canonical(x) for x in rule["lines"]), n, ops
                )
            )
        if len({x.name for x in limits}) != len(limits):
            raise ValueError("duplicate action limit name")
        return cls(tuple(gates), tuple(limits))
