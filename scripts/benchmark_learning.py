"""Measure the same pure-relative learner on CPU and CPU-actor/GPU configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fzd_shunting.data import write_json
from fzd_shunting.domain import Rules, Yard
from fzd_shunting.learning import load_dataset, split_dataset, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="data/point_to_area/augmented")
    parser.add_argument("--rules", default="configs/dispatch.json")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-hooks", type=int, default=40)
    parser.add_argument("--groups-per-batch", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--batch-wait-ms", type=float, default=15)
    parser.add_argument("--output", default="runs/throughput/summary.json")
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=["cpu", "gpu4", "gpu8"],
        default=["cpu", "gpu4", "gpu8"],
    )
    args = parser.parse_args()
    ds, records = load_dataset(args.requests, Yard.load(), Rules.load(args.rules))
    ds, _, _, split = split_dataset(ds)
    results = []
    for name, device, precision, workers in [
        ("cpu", "cpu", "float32", 0),
        ("gpu4", "cuda", "bfloat16", 4),
        ("gpu8", "cuda", "bfloat16", 8),
    ]:
        if name not in args.configurations:
            continue
        samples = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                line = (
                    subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    .stdout.strip()
                    .splitlines()[0]
                )
                samples.append(tuple(float(x.strip()) for x in line.split(",")))
                stop.wait(0.5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.perf_counter()
        try:
            result = train(
                ds,
                iterations=args.iterations,
                max_hooks=args.max_hooks,
                group_size=args.group_size,
                groups_per_batch=args.groups_per_batch,
                minibatch_size=args.minibatch_size,
                batch_wait_ms=args.batch_wait_ms,
                workers=workers,
                device=device,
                precision=precision,
                output=Path(args.output).parent / (name + ".pt"),
                progress=lambda row: print(
                    json.dumps({"configuration": name, **row}), flush=True
                ),
            )
        finally:
            stop.set()
            thread.join()
        rows = [
            json.loads(x)
            for x in (Path(args.output).parent / (name + ".history.jsonl"))
            .read_text()
            .splitlines()
        ]
        steady = rows[1:] if len(rows) > 1 else rows
        report = {
            "name": name,
            "runtime": result["runtime"],
            "wall_seconds_including_startup": time.perf_counter() - started,
            "environment_steps": result["environment_steps"],
            "steady_steps_per_second": sum(x["environment_steps"] for x in steady)
            / sum(x["iteration_seconds"] for x in steady),
            "steady_rollout_steps_per_second": sum(
                x["environment_steps"] for x in steady
            )
            / sum(x["sampling_seconds"] for x in steady),
            "mean_inference_batch": statistics.mean(
                x["mean_inference_batch"] for x in steady
            ),
            "gpu_utilization_mean_including_startup": (
                statistics.mean(x[0] for x in samples) if samples else None
            ),
            "gpu_utilization_max": max((x[0] for x in samples), default=None),
            "gpu_memory_peak_mb_observed": max((x[1] for x in samples), default=None),
        }
        results.append(report)
        write_json(
            args.output,
            {
                "configuration": vars(args),
                "dataset_split": split,
                "training_requests": len(ds),
                "results": results,
                "scope": "Same network, supplied dataset, group size, global hook ceiling and update epochs; stochastic trajectories differ across devices/schedules. Throughput excludes first iteration; GPU monitoring includes startup and display load.",
            },
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
