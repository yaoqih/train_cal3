"""Copy, normalize and reproducibly add synthetic relative-position constraints."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .domain import Yard
from .environment import Environment
from .request import load_request, normalize_request
from .goals import slot_bounds


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def slot_witness(scenario, yard, rng):
    """Bipartite matching witnesses target-slot consistency, not route feasibility."""
    capacities = slot_bounds(scenario, yard)
    domains = {}
    cars = scenario.cars
    for car in scenario.vehicles:
        slots = [
            (t.line, p)
            for t in car.targets
            for p in (t.positions or range(1, capacities[t.line] + 1))
            if p <= capacities[t.line]
        ]
        if car.no in scenario.protected:
            slots = [(car.initial_line, car.initial_position)]
        rng.shuffle(slots)
        initial = (car.initial_line, car.initial_position)
        if initial in slots:
            slots.remove(initial)
            slots.insert(0, initial)
        domains[car.no] = slots
    owner = {}

    def assign(no, seen):
        for slot in domains[no]:
            if slot in seen:
                continue
            seen.add(slot)
            if slot not in owner or assign(owner[slot], seen):
                owner[slot] = no
                return True
        return False

    order = sorted(cars, key=lambda no: (len(domains[no]), no))
    if not all(assign(no, set()) for no in order):
        return None
    return {no: slot for slot, no in owner.items()}


def augment_request(normalized, yard, seed, fraction=0.2):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    rng = random.Random(seed)
    scenario = load_request(normalized, yard)
    witness = slot_witness(scenario, yard, rng)
    data = copy.deepcopy(normalized)
    capacity = slot_bounds(scenario, yard)
    constraints = 0
    selected = 0
    for row in data["StartStatus"]:
        if witness is None or rng.random() >= fraction:
            continue
        changed = False
        witness_line, witness_position = witness[row["No"]]
        for line, target in row["TargetLines"].items():
            if target.get("ForceTargetPosition"):
                continue
            domain = list(range(1, capacity[line] + 1))
            # Do not accidentally create an anchor inconsistent with the witness.
            if (
                line in dict(scenario.inner_capacities)
                and line == row["Line"]
                and ((witness_line, witness_position) != (line, row["Position"]))
            ):
                domain = [p for p in domain if p != row["Position"]]
            if not domain:
                continue
            chosen = (
                {witness_position} if line == witness_line else {rng.choice(domain)}
            )
            if rng.random() < 0.3:
                chosen.add(rng.choice(domain))
            target["ForceTargetPosition"] = sorted(chosen)
            constraints += 1
            changed = True
        selected += int(changed)
    # A protected vehicle also makes the whole suffix behind it inaccessible.
    # Relax only newly synthesized constraints if they conflict with that fixed order.
    from .dispatch import Dispatcher

    constrained = load_request(data, yard)
    dispatcher = Dispatcher(Environment(yard, constrained))
    originals = {row["No"]: row for row in normalized["StartStatus"]}
    augmented_rows = {row["No"]: row for row in data["StartStatus"]}
    relaxed = 0
    for line in dispatcher.protected_lines:
        stack = constrained.initial.stack(line)
        first = min(i for i, no in enumerate(stack) if no in dispatcher.protected)
        if dispatcher.order_assignment(stack[first:], line) is not None:
            continue
        for no in stack[first:]:
            target = augmented_rows[no]["TargetLines"].get(line)
            original = originals[no]["TargetLines"].get(line, {})
            if (
                target
                and target.get("ForceTargetPosition")
                and not original.get("ForceTargetPosition")
            ):
                target.pop("ForceTargetPosition")
                relaxed += 1
    constraints = sum(
        bool(target.get("ForceTargetPosition"))
        and not originals[row["No"]]["TargetLines"][line].get("ForceTargetPosition")
        for row in data["StartStatus"]
        for line, target in row["TargetLines"].items()
    )
    selected = sum(
        any(
            bool(target.get("ForceTargetPosition"))
            and not originals[row["No"]]["TargetLines"][line].get("ForceTargetPosition")
            for line, target in row["TargetLines"].items()
        )
        for row in data["StartStatus"]
    )
    lengths = Counter()
    if witness:
        for no, (line, _) in witness.items():
            lengths[line] += scenario.cars[no].length_mm
    length_issues = [
        line
        for line, length in lengths.items()
        if length > yard.by_name[line].final_capacity_mm
    ]
    metadata = {
        "synthetic": True,
        "seed": seed,
        "fraction": fraction,
        "vehicles_augmented": selected,
        "constraints_added": constraints,
        "relaxed_synthetic_constraints": relaxed,
        "augmentation_version": 1,
        "slot_matching_found": witness is not None,
        "witness_length_issues": length_issues,
        "scope": "Target-slot matching only; not a shunting plan or a solvability guarantee.",
    }
    data["SyntheticConstraints"] = metadata
    return data, metadata


def prepare_data(source, destination, yard, seed=20260924, fraction=0.2):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError(
            "destination must be separate from the read-only source directory"
        )
    files = sorted(source.glob("*.json"))
    if not files:
        raise ValueError("no JSON requests found")
    manifest = {"seed": seed, "fraction": fraction, "source": str(source), "files": []}
    for file in files:
        raw_bytes = file.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        raw_target = destination / "raw" / file.name
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        if raw_target.exists() and raw_target.read_bytes() != raw_bytes:
            raise ValueError(
                "refusing to replace a different raw copy: " + str(raw_target)
            )
        shutil.copyfile(file, raw_target)
        raw = json.loads(raw_bytes)
        normalized = normalize_request(raw)
        write_json(destination / "normalized" / file.name, normalized)
        file_seed = int.from_bytes(
            hashlib.sha256((str(seed) + ":" + file.name).encode()).digest()[:8], "big"
        )
        augmented, metadata = augment_request(normalized, yard, file_seed, fraction)
        write_json(destination / "augmented" / file.name, augmented)
        scenario = load_request(augmented, yard)
        env = Environment(yard, scenario)
        manifest["files"].append(
            {
                "file": file.name,
                "source_sha256": digest,
                **metadata,
                "vehicles": len(scenario.vehicles),
                "protected": len(scenario.protected),
                "initial_state_errors": env.validate_state(scenario.initial),
            }
        )
    write_json(destination / "manifest.json", manifest)
    return manifest
