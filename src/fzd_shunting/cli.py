from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare_data, write_json
from .dispatch import Dispatcher
from .domain import Rules, Yard
from .environment import Environment
from .planning import plan, replay, rollout, next_hook
from .request import load_request


def make_dispatcher(request, yard, rules):
    return Dispatcher(Environment(yard, load_request(request, yard)), rules)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="福州东：固定规则训练权重、按当前状态逐勾决策"
    )
    parser.add_argument("--yard", help="canonical yard JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare-data", help="copy/augment canonical train_cal3 requests"
    )
    prep.add_argument("source")
    prep.add_argument("destination")
    prep.add_argument("--seed", type=int, default=20260924)
    prep.add_argument("--fraction", type=float, default=0.2)
    audit = commands.add_parser("audit")
    audit.add_argument("requests")
    audit.add_argument("--output")
    audit.add_argument("--candidates", action="store_true")
    audit.add_argument("--rules")
    for name in ["plan", "next", "replay", "train", "evaluate"]:
        cmd = commands.add_parser(name)
        cmd.add_argument("request")
        cmd.add_argument("--rules")
        if name in ("plan", "next", "evaluate"):
            cmd.add_argument(
                "--checkpoint",
                required=True,
                help="matching pure-relative schema-4 weights",
            )
        if name in ("plan", "next"):
            cmd.add_argument(
                "--current-state",
                help="versioned observed snapshot for the original request",
            )
        if name in ("plan", "next", "evaluate", "train"):
            cmd.add_argument(
                "--max-hooks",
                type=int,
                default=64,
                help="total-plan hook ceiling, including already executed hooks",
            )
        if name == "plan":
            cmd.add_argument("--mode", choices=["greedy", "search"], default="greedy")
            cmd.add_argument("--device", default="cuda")
            cmd.add_argument(
                "--precision", choices=["float32", "bfloat16"], default="bfloat16"
            )
            cmd.add_argument("--workers", type=int, default=4)
            cmd.add_argument("--inference-batch-size", type=int, default=32)
            cmd.add_argument("--expansions", type=int, default=5000)
            cmd.add_argument("--branch-limit", type=int, default=40)
            cmd.add_argument("--seconds", type=float, default=180)
            cmd.add_argument("--frontier", type=int, default=4000)
            cmd.add_argument("--states", type=int, default=50000)
            cmd.add_argument("--output", default="runs/plan.json")
        elif name == "next":
            cmd.add_argument("--device", default="cpu")
            cmd.add_argument("--output", default="runs/next-hook.json")
        elif name == "replay":
            cmd.add_argument("plan")
        elif name == "train":
            cmd.add_argument("--iterations", type=int, default=1000)
            cmd.add_argument("--group-size", type=int, default=8)
            cmd.add_argument("--groups-per-batch", type=int, default=16)
            cmd.add_argument("--update-epochs", type=int, default=2)
            cmd.add_argument("--minibatch-size", type=int, default=128)
            cmd.add_argument("--trajectory-batch-size", type=int, default=32)
            cmd.add_argument("--branch-fraction", type=float, default=0.3)
            cmd.add_argument("--state-pool-size", type=int, default=2048)
            cmd.add_argument("--workers", type=int, default=8)
            cmd.add_argument("--device", default="cuda")
            cmd.add_argument(
                "--precision", choices=["float32", "bfloat16"], default="bfloat16"
            )
            cmd.add_argument("--cpu-threads", type=int, default=1)
            cmd.add_argument("--inference-batch-size", type=int, default=64)
            cmd.add_argument("--batch-wait-ms", type=float, default=15)
            cmd.add_argument("--hidden", type=int, default=96)
            cmd.add_argument("--layers", type=int, default=3)
            cmd.add_argument("--learning-rate", type=float, default=3e-4)
            cmd.add_argument("--entropy-weight", type=float, default=0.01)
            cmd.add_argument("--clip-ratio", type=float, default=0.2)
            cmd.add_argument("--target-kl", type=float, default=0.03)
            cmd.add_argument("--validate-every", type=int, default=25)
            cmd.add_argument("--checkpoint-every", type=int, default=10)
            cmd.add_argument("--validation-fraction", type=float, default=0.15)
            cmd.add_argument("--test-fraction", type=float, default=0.1)
            cmd.add_argument("--seed", type=int, default=7)
            cmd.add_argument("--resume")
            cmd.add_argument("--output", default="runs/policy.pt")
        elif name == "evaluate":
            cmd.add_argument("--device", default="cuda")
            cmd.add_argument(
                "--precision", choices=["float32", "bfloat16"], default="float32"
            )
            cmd.add_argument("--workers", type=int, default=8)
            cmd.add_argument("--samples-per-request", type=int, default=8)
            cmd.add_argument("--output", default="runs/evaluation.json")
    args = parser.parse_args(argv)
    yard = Yard.load(args.yard)
    if args.command == "prepare-data":
        result = prepare_data(
            args.source, args.destination, yard, args.seed, args.fraction
        )
        print(
            json.dumps(
                {
                    "requests": len(result["files"]),
                    "manifest": str(Path(args.destination) / "manifest.json"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    rules = Rules.load(args.rules)
    if args.command == "audit":
        from .feasibility import terminal_feasibility

        path = Path(args.requests)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not files:
            raise ValueError("no requests found")
        rows = []
        for file in files:
            try:
                d = make_dispatcher(file, yard, rules)
                errors = d.validate(d.state, d.context)
                row = {
                    "file": file.name,
                    "vehicles": len(d.env.cars),
                    "protected": len(d.protected),
                    "errors": errors,
                }
                if not errors:
                    row["terminal_check"] = terminal_feasibility(d)
                if args.candidates and not errors:
                    reasons = {}
                    row["candidates"] = len(d.candidates(d.state, diagnostics=reasons))
                    row["rejections"] = reasons
            except (ValueError, KeyError, TypeError) as exc:
                row = {"file": file.name, "errors": [str(exc)]}
            rows.append(row)
        result = {
            "requests": len(rows),
            "initial_valid": sum(not r["errors"] for r in rows),
            "terminal_infeasible": sum(
                r.get("terminal_check", {}).get("status") == "infeasible" for r in rows
            ),
            "valid": sum(
                not r["errors"] and r["terminal_check"]["status"] == "not_disproved"
                for r in rows
            ),
            "scope": "Initial legality and necessary terminal conditions; passing is not a complete-route proof.",
            "files": rows,
        }
        if args.output:
            write_json(args.output, result)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "files"}, ensure_ascii=False
            )
        )
        return 0 if result["valid"] == result["requests"] else 2
    if args.command in ("train", "evaluate"):
        from .learning import train, evaluate, load_dataset, split_dataset
        from .policy.gnn import load_model

        requests, records = load_dataset(args.request, yard, rules)
        if args.command == "train":
            training, validation, test, split = split_dataset(
                requests, args.validation_fraction, args.test_fraction
            )
            result = train(
                training,
                args.iterations,
                args.max_hooks,
                args.seed,
                args.output,
                group_size=args.group_size,
                update_epochs=args.update_epochs,
                groups_per_batch=args.groups_per_batch,
                minibatch_size=args.minibatch_size,
                trajectory_batch_size=args.trajectory_batch_size,
                branch_fraction=args.branch_fraction,
                state_pool_size=args.state_pool_size,
                workers=args.workers,
                device=args.device,
                precision=args.precision,
                cpu_threads=args.cpu_threads,
                inference_batch_size=args.inference_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                hidden=args.hidden,
                layers=args.layers,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                clip_ratio=args.clip_ratio,
                target_kl=args.target_kl,
                validate_every=args.validate_every,
                checkpoint_every=args.checkpoint_every,
                progress=lambda row: print(
                    json.dumps(row, ensure_ascii=False), flush=True
                ),
                validation=validation,
                resume=args.resume,
                dataset_metadata={
                    "files": records,
                    "split": split,
                    "test_requests": len(test),
                },
            )
        else:
            import torch

            torch.set_num_threads(1)
            result = evaluate(
                load_model(args.checkpoint, args.device),
                requests,
                args.max_hooks,
                workers=args.workers,
                precision=args.precision,
                samples_per_request=args.samples_per_request,
            )
            result["dataset"] = records
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    d = make_dispatcher(args.request, yard, rules)
    if args.command == "replay":
        result = json.loads(Path(args.plan).read_text())
        if result.get("schema_version") != 2:
            raise ValueError("expected train_cal3 plan schema 2")
        d.restore(result["initial_snapshot"])
        start, context = d.state, d.context
        state = replay(
            d, result["actions"], result["events"], start=start, context=context
        )
        if result["final_snapshot"] != json.loads(json.dumps(d.snapshot())):
            raise ValueError("plan final snapshot does not match replay")
        done = d.done(state)
        if result["status"] == "complete" and not done:
            raise ValueError("plan claims completion but fails validation")
        print(
            json.dumps(
                {
                    "replayed_hooks": len(result["actions"]),
                    "total_hooks": state.hook,
                    "complete": done,
                    "goal_errors": d.goal_errors(state),
                },
                ensure_ascii=False,
            )
        )
        return 0
    from .policy.gnn import load_model

    import torch

    torch.set_num_threads(1)
    model = load_model(args.checkpoint, args.device)
    model.hook_budget = args.max_hooks
    model.validate_profile(d)
    if args.current_state:
        d.restore(json.loads(Path(args.current_state).read_text()))
    if args.command == "next":
        result = next_hook(d, model, args.max_hooks)
    elif args.mode == "search":
        result = plan(
            d,
            args.expansions,
            args.max_hooks,
            args.branch_limit,
            policy=model,
            inference_batch_size=args.inference_batch_size,
            workers=args.workers,
            precision=args.precision,
            time_limit=args.seconds,
            max_frontier=args.frontier,
            max_states=args.states,
            start=d.state,
            context=d.context,
        )
    else:
        result = rollout(d, model, args.max_hooks, start=d.state, context=d.context)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "actions",
                    "events",
                    "initial_snapshot",
                    "final_snapshot",
                    "current_snapshot",
                    "predicted_snapshot",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] in ("complete", "action") else 2
