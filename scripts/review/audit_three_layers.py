"""Reproduce current-contract invariants and held-out policy measurements."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from fzd_shunting.cli import make_dispatcher
from fzd_shunting.data import write_json
from fzd_shunting.dispatch import Dispatcher
from fzd_shunting.domain import Action, Rules, Target, Track, Yard
from fzd_shunting.environment import Environment
from fzd_shunting.learning import evaluate, load_dataset, split_dataset
from fzd_shunting.planning import next_hook, plan, replay
from fzd_shunting.policy.gnn import load_model
from fzd_shunting.request import load_request


def row(no, line, pos=1, targets=None):
    return {
        "No": no,
        "Line": line,
        "Position": pos,
        "Length": 13.2,
        "TargetLines": {"B": {}} if targets is None else targets,
    }


def small_yard(rows):
    yard = Yard(
        (
            Track("R", 0, 0, "transit"),
            Track("A", 300000),
            Track("B", 300000),
            Track("C", 300000),
        ),
        (("A", "R"), ("B", "R"), ("C", "R")),
    )
    return Dispatcher(
        Environment(
            yard,
            load_request(
                {"StartStatus": rows, "locoNode": {"Line": "R", "End": "North"}}, yard
            ),
        )
    )


def relative_order_oracle(trials=1000):
    rng = random.Random(9341)
    for _ in range(trials):
        domains = [
            tuple(sorted(rng.sample(range(1, 7), rng.randint(1, 4))))
            for _ in range(rng.randint(1, 5))
        ]
        d = small_yard(
            [
                row(str(i), "B", i + 1, {"B": {"ForceTargetPosition": list(p)}})
                for i, p in enumerate(domains)
            ]
        )
        expected = any(
            all(a < b for a, b in zip(ps, ps[1:])) for ps in itertools.product(*domains)
        )
        assert (
            d.order_assignment(tuple(str(i) for i in range(len(domains))), "B")
            is not None
        ) == expected
    return {"checks": trials, "mismatch": None}


def transition_invariants(steps=250):
    d = small_yard(
        [
            row(str(i), "A" if i < 3 else "B", i % 3 + 1, {"A": {}, "B": {}, "C": {}})
            for i in range(6)
        ]
    )
    rng = random.Random(21)
    state = d.state
    for _ in range(steps):
        c = rng.choice(d.candidates(state))
        reverse = Action(
            c.action.line,
            "put" if c.action.operation == "get" else "get",
            c.action.count,
        )
        back, _, _ = d.env.transition(c.after, reverse)
        assert back.stacks == state.stacks and back.train == state.train
        state = d.step(state, c)
        assert not d.validate(state)
    return {
        "transitions": steps,
        "inventory_and_order": "passed",
        "scope": "simple unprotected yard; inverse check excludes locomotive location and hook",
    }


def benchmark(dispatchers, model, expansions, seconds):
    sizes = sorted(dispatchers, key=lambda d: len(d.env.cars))
    results = []
    for d in (sizes[0], sizes[len(sizes) // 2], sizes[-1]):
        state = d.env.scenario.initial
        timings = []
        for _ in range(5):
            t = time.perf_counter()
            candidates = d.candidates(state)
            timings.append(time.perf_counter() - t)
        t = time.perf_counter()
        for _ in range(5):
            model.log_probabilities(d, state, candidates)
        neural = (time.perf_counter() - t) / 5
        answer = plan(
            d,
            max_expansions=expansions,
            max_hooks=40,
            time_limit=seconds,
            policy=model,
        )
        replay(d, answer["actions"], answer["events"], start=state)
        item = {
            "request": d.env.scenario.name,
            "vehicles": len(d.env.cars),
            "candidates": len(candidates),
            "candidate_median_ms": statistics.median(timings) * 1000,
            "gnn_cpu_ms": neural * 1000,
            "search_seconds": answer["elapsed_seconds"],
            "status": answer["status"],
            "total_hooks": answer["total_hooks"],
            "completed_vehicles": answer["completed_vehicles"],
            "expanded_states": answer["expanded_states"],
            "stop_reason": answer["stop_reason"],
            "strict_replay": "passed",
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rules", default="configs/dispatch.json")
    parser.add_argument("--expansions", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--output", default="runs/architecture-review/current.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_model(args.checkpoint)
    template = make_dispatcher(
        ROOT / "scenarios/demo.json", Yard.load(), Rules.load(args.rules)
    )
    requests, records = load_dataset(
        ROOT / "data/point_to_area/augmented", template.env.yard, template.rules
    )
    train, validation, test, split = split_dataset(requests)
    result = {
        "checkpoint": args.checkpoint,
        "profile": model.profile,
        "relative_order_oracle": relative_order_oracle(),
        "transition_invariants": transition_invariants(),
        "validation_policy": evaluate(model, validation, 40),
        "dataset": {
            "usable": len(requests),
            "training": len(train),
            "validation": len(validation),
            "test": len(test),
            "split": split,
            "records": records,
        },
        "real_request_benchmark": benchmark(
            validation, model, args.expansions, args.seconds
        ),
        "scope": "Validation partition only; independent test requests are reserved for scripts/validate_learning.py.",
    }
    write_json(args.output, result)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
