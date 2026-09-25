"""Evaluate untouched date-held-out requests, bounded search and exact replay."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fzd_shunting.learning import load_dataset, split_dataset, evaluate
from fzd_shunting.domain import Yard, Rules
from fzd_shunting.policy.gnn import load_model
from fzd_shunting.planning import plan, replay
from fzd_shunting.data import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--requests", default="data/point_to_area/augmented")
    p.add_argument("--rules", default="configs/dispatch.json")
    p.add_argument("--checkpoint", default="runs/policy.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-hooks", type=int, default=40)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--expansions", type=int, default=5000)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--output", default="runs/learning-validation.json")
    args = p.parse_args()
    torch.set_num_threads(1)
    ds, records = load_dataset(args.requests, Yard.load(), Rules.load(args.rules))
    training, validation, test, split = split_dataset(
        ds, args.validation_fraction, args.test_fraction
    )
    if not test:
        raise ValueError("no independent date-held-out test set")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    observed = {d.env.scenario_hash for d in test}
    used = set(checkpoint["training_config"]["requests"]) | set(
        checkpoint["training_config"]["validation"]
    )
    if observed & used:
        raise ValueError("test requests overlap checkpoint training or validation")
    m = load_model(args.checkpoint, args.device)
    result = dict(
        schema_version=1,
        checkpoint=args.checkpoint,
        parameters=sum(p.numel() for p in m.parameters()),
        config=vars(args),
        dataset=records,
        split=split,
        training_iterations=checkpoint["iterations_completed"],
        training_steps=checkpoint["environment_steps"],
        policy=evaluate(
            m, test, args.max_hooks, workers=args.workers, precision=args.precision
        ),
        search=[],
    )
    directory = Path(args.output).with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    for d in test:
        print(
            json.dumps(
                dict(phase="search", request=d.env.scenario.name), ensure_ascii=False
            ),
            flush=True,
        )
        started = time.perf_counter()
        output = plan(
            d,
            policy=m,
            max_hooks=args.max_hooks,
            max_expansions=args.expansions,
            time_limit=args.seconds,
            max_frontier=8000,
            max_states=200000,
            workers=args.workers,
            inference_batch_size=32,
            precision=args.precision,
        )
        path = directory / (d.env.scenario.name + ".plan.json")
        write_json(path, output)
        saved = json.loads(path.read_text())
        actual = replay(d, saved["actions"], saved["events"])
        assert json.loads(json.dumps(d.snapshot())) == saved["final_snapshot"]
        assert (saved["status"] == "complete") == d.done(actual)
        row = dict(
            request=d.env.scenario.name,
            complete=output["status"] == "complete",
            total_hooks=output["total_hooks"],
            elapsed_seconds=time.perf_counter() - started,
            stop_reason=output["stop_reason"],
            expanded_states=output["expanded_states"],
            inference=output["inference"],
            strict_replay=True,
            plan=str(path),
        )
        result["search"].append(row)
        write_json(args.output, result)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result["search_complete"] = sum(r["complete"] for r in result["search"])
    result["scope"] = (
        "Independent held-out dates; passing replay is not completion or optimality. Test set must not be used for further model selection."
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
