"""Pure group-relative policy learning on supplied requests; no teacher or curriculum."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import write_json
from .dispatch import Dispatcher
from .domain import BusinessContext, digest
from .environment import Environment
from .objective import trajectory_score
from .policy.gnn import (
    CHECKPOINT_SCHEMA,
    FEATURE_SCHEMA,
    GraphPolicy,
    collate_graphs,
    load_model,
)
from .request import load_request
from .sampling import ParallelSampler, StartPoint, StatePool
from .feasibility import terminal_feasibility
from collections import Counter

ALGORITHM = "grouped-root-fork-relative-v2"


def group_advantages(scores):
    values = torch.tensor(scores, dtype=torch.float32)
    spread = values.std(unbiased=False)
    return (
        (values - values.mean()) / spread.clamp_min(1e-3)
        if float(spread) > 1e-8
        else torch.zeros_like(values)
    )


def load_dataset(path, yard, rules):
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise ValueError("no request JSON files")
    dispatchers, records = [], []
    for file in files:
        entry = {
            "file": str(file),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
        try:
            d = Dispatcher(Environment(yard, load_request(file, yard)), rules)
            errors = d.validate(d.env.scenario.initial, BusinessContext())
            if errors:
                raise ValueError("; ".join(errors))
            entry["terminal_check"] = terminal_feasibility(d)
            if entry["terminal_check"]["status"] == "infeasible":
                raise ValueError(
                    "INFEASIBLE_TARGETS: "
                    + "; ".join(entry["terminal_check"]["reasons"])
                )
            if d.done(d.env.scenario.initial, BusinessContext()):
                entry["status"] = "already_complete"
            else:
                entry.update(status="usable", scenario_hash=d.env.scenario_hash)
                dispatchers.append(d)
        except (ValueError, KeyError, TypeError) as exc:
            if not path.is_dir():
                raise
            entry.update(status="invalid", error=str(exc))
        records.append(entry)
    if not dispatchers:
        raise ValueError("no valid nontrivial requests in dataset")
    return dispatchers, records


def split_dataset(dispatchers, validation_fraction=0.15, test_fraction=0.1):
    """Keep all variants of the same source request/date-week in one partition."""
    import datetime
    import re

    if (
        min(validation_fraction, test_fraction) < 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("validation fraction must be in [0,1)")
    groups = {}
    for d in dispatchers:
        match = re.search(r"(20\d{6})", d.env.scenario.name)
        if match:
            day = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            year, week, _ = day.isocalendar()
            family = f"{year}-W{week:02d}"
        else:
            family = d.env.scenario.name or d.env.scenario_hash
        groups.setdefault(family, []).append(d)
    keys = sorted(groups)
    test_n = (
        min(len(keys) - 1, max(1, round(len(keys) * test_fraction)))
        if test_fraction and len(keys) > 2
        else 0
    )
    val_n = (
        min(len(keys) - test_n - 1, max(1, round(len(keys) * validation_fraction)))
        if validation_fraction and len(keys) > 1
        else 0
    )
    test_keys = set(keys[-test_n:]) if test_n else set()
    validation_keys = (
        set(keys[len(keys) - test_n - val_n : len(keys) - test_n]) if val_n else set()
    )
    training_keys = set(keys) - test_keys - validation_keys

    def partition(selected):
        return [d for k in keys if k in selected for d in groups[k]]

    return (
        partition(training_keys),
        partition(validation_keys),
        partition(test_keys),
        dict(
            training_families=sorted(training_keys),
            validation_families=sorted(validation_keys),
            test_families=sorted(test_keys),
            test_requests=[d.env.scenario.name for d in partition(test_keys)],
        ),
    )


def evaluate(
    model,
    dispatchers,
    max_hooks=64,
    *,
    workers=0,
    precision="float32",
    sampler=None,
    request_indices=None,
    samples_per_request=8,
    seed=1701,
):
    dispatchers = list(dispatchers)
    if samples_per_request < 1:
        raise ValueError("positive evaluation sample count required")
    if not dispatchers:
        return dict(
            requests=0,
            complete=0,
            completion_rate=None,
            mean_total_hooks_on_complete=None,
            results=[],
        )
    owned = sampler is None
    if owned:
        sampler = ParallelSampler(dispatchers, workers=workers)
    indices = (
        list(range(len(dispatchers))) if request_indices is None else request_indices
    )
    device = next(model.parameters()).device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        # Evaluation neither consumes training random state nor fills the branch pool.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            trajectories, timing = sampler.collect(
                model,
                indices,
                1,
                max_hooks,
                precision=precision,
                greedy=True,
                record=False,
            )
            sampled, sample_timing = sampler.collect(
                model,
                indices,
                samples_per_request,
                max_hooks,
                precision=precision,
                record=False,
            )
    finally:
        if owned:
            sampler.close()
    rows, sample_rows = [], []
    for i, (d, trajectory) in enumerate(zip(dispatchers, trajectories)):
        outcome = trajectory.outcome
        rows.append(
            dict(
                request=d.env.scenario.name,
                scenario_hash=d.env.scenario_hash,
                status=outcome.stop_reason,
                hooks=outcome.final_state.hook,
                complete=outcome.complete,
                complete_within_40=outcome.complete and outcome.final_state.hook <= 40,
                vehicles=len(d.env.cars),
                completed_vehicles=d.progress(outcome.final_state),
            )
        )
        group = sampled[i * samples_per_request : (i + 1) * samples_per_request]
        hooks = [t.outcome.final_state.hook for t in group if t.outcome.complete]
        sample_rows.append(
            dict(
                request=d.env.scenario.name,
                complete=bool(hooks),
                complete_within_40=bool(hooks) and min(hooks) <= 40,
                best_total_hooks=min(hooks) if hooks else None,
                stops=dict(Counter(t.outcome.stop_reason for t in group)),
            )
        )
    successes = [r for r in rows if r["complete"]]
    sampled_successes = [r for r in sample_rows if r["complete"]]
    return dict(
        requests=len(rows),
        complete=len(successes),
        completion_rate=len(successes) / len(rows),
        complete_within_40=sum(r["complete_within_40"] for r in rows),
        mean_total_hooks_on_complete=(
            sum(r["hooks"] for r in successes) / len(successes) if successes else None
        ),
        stop_counts=dict(Counter(r["status"] for r in rows)),
        timing=timing,
        results=rows,
        sampled=dict(
            samples_per_request=samples_per_request,
            complete=len(sampled_successes),
            completion_rate=len(sampled_successes) / len(rows),
            complete_within_40=sum(r["complete_within_40"] for r in sample_rows),
            mean_best_total_hooks=(
                sum(r["best_total_hooks"] for r in sampled_successes)
                / len(sampled_successes)
                if sampled_successes
                else None
            ),
            timing=sample_timing,
            results=sample_rows,
        ),
    )


def update_policy(
    model,
    optimizer,
    trajectories,
    *,
    group_size,
    update_epochs,
    minibatch_size,
    clip_ratio,
    entropy_weight,
    target_kl,
    precision,
    rng,
    progress=None,
    trajectory_batch_size=32,
):
    """Whole-group advantages, trajectory minibatches, first-decision credit for forks."""
    if trajectory_batch_size < group_size:
        raise ValueError(
            "trajectory batch must contain a whole number of request groups"
        )
    groups, records_by_trajectory = [], []
    for start in range(0, len(trajectories), group_size):
        group = trajectories[start : start + group_size]
        if (
            len(group) != group_size
            or len({(t.group_index, t.request_index, t.start_key) for t in group}) != 1
        ):
            raise ValueError(
                "relative advantages require a complete same-state request group"
            )
        advantages = group_advantages([t.outcome.score for t in group]).tolist()
        groups.append(
            dict(
                request_index=group[0].request_index,
                start_hook=group[0].start_hook,
                scores=[t.outcome.score for t in group],
                advantages=advantages,
                total_hooks=[t.outcome.final_state.hook for t in group],
                stops=[t.outcome.stop_reason for t in group],
                complete=sum(t.outcome.complete for t in group),
                complete_within_40=sum(
                    t.outcome.complete and t.outcome.final_state.hook <= 40
                    for t in group
                ),
                improved_from_start=sum(
                    t.outcome.score > t.outcome.start_score for t in group
                ),
                has_relative_signal=any(abs(a) > 1e-8 for a in advantages),
            )
        )
        for t, advantage in zip(group, advantages):
            credited = t.samples[:1] if t.start_hook else t.samples
            records_by_trajectory.append([(sample, advantage) for sample in credited])
    n = sum(map(len, records_by_trajectory))
    device = next(model.parameters()).device
    model.train()
    updates, summaries, halted = 0, [], False
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = last_progress = time.perf_counter()
    groups_per_update = trajectory_batch_size // group_size
    for epoch in range(update_epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for first_group in range(0, len(order), groups_per_update):
            selected = order[first_group : first_group + groups_per_update]
            entries = [
                item
                for g in selected
                for i in range(g * group_size, (g + 1) * group_size)
                for item in records_by_trajectory[i]
            ]
            if not entries:
                continue
            rng.shuffle(entries)
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(4, device=device)
            for first in range(0, len(entries), minibatch_size):
                records = entries[first : first + minibatch_size]
                batch = collate_graphs([s.graph for s, _ in records])
                old = np.zeros(batch.action_mask.shape, dtype=np.float32)
                for i, (sample, _) in enumerate(records):
                    old[i, : len(sample.old_log_probs)] = sample.old_log_probs
                if device.type == "cuda":
                    batch = batch.pin_memory()
                batch = batch.to(device)
                old = torch.from_numpy(old).to(device)
                actions = torch.tensor(
                    [s.action_index for s, _ in records], device=device
                )
                advantages = torch.tensor([a for _, a in records], device=device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bfloat16",
                ):
                    logp = torch.log_softmax(
                        model.forward_batch(batch).float(), dim=1
                    ).masked_fill(~batch.action_mask, 0)
                    ratio = (
                        (
                            logp.gather(1, actions[:, None])
                            - old.gather(1, actions[:, None])
                        )
                        .squeeze(1)
                        .exp()
                    )
                    surrogate = torch.minimum(
                        ratio * advantages,
                        ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages,
                    )
                    entropy = -(logp.exp() * batch.action_mask * logp).sum(1)
                    kl = (old.exp() * batch.action_mask * (old - logp)).sum(1)
                    loss = -surrogate.sum() / (
                        len(selected) * group_size
                    ) - entropy_weight * entropy.sum() / len(entries)
                loss.backward()
                totals += torch.stack(
                    (
                        loss.detach(),
                        kl.detach().sum() / len(entries),
                        entropy.detach().sum() / len(entries),
                        ((ratio.detach() - 1).abs() > clip_ratio).float().sum()
                        / len(entries),
                    )
                )
                if progress and time.perf_counter() - last_progress >= 15:
                    progress(
                        dict(
                            phase="updating",
                            epoch=epoch + 1,
                            trajectory_block=first_group // groups_per_update + 1,
                            optimizer_updates=updates,
                        )
                    )
                    last_progress = time.perf_counter()
            loss_value, kl_value, entropy_value, clipped = (
                totals.detach().cpu().tolist()
            )
            if not all(
                np.isfinite(v) for v in (loss_value, kl_value, entropy_value, clipped)
            ):
                raise RuntimeError("non-finite relative policy update")
            if updates and target_kl and kl_value > target_kl:
                optimizer.zero_grad(set_to_none=True)
                summaries.append(dict(epoch=epoch + 1, early_stop_kl=kl_value))
                halted = True
                break
            grad = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
            )
            optimizer.step()
            updates += 1
            summaries.append(
                dict(
                    epoch=epoch + 1,
                    loss=loss_value,
                    old_policy_kl=kl_value,
                    entropy=entropy_value,
                    clip_fraction=clipped,
                    gradient_norm=grad,
                    trajectories=len(selected) * group_size,
                    decisions=len(entries),
                )
            )
        if halted:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return dict(
        groups=groups,
        optimizer_updates=updates,
        update_epochs=summaries,
        training_seconds=time.perf_counter() - started,
        decision_samples=n,
    )


def train(
    dispatchers,
    iterations=1000,
    max_hooks=64,
    seed=7,
    output="runs/policy.pt",
    *,
    group_size=8,
    groups_per_batch=16,
    update_epochs=2,
    minibatch_size=128,
    trajectory_batch_size=32,
    branch_fraction=0.3,
    state_pool_size=2048,
    workers=8,
    device="cuda",
    precision="bfloat16",
    inference_batch_size=64,
    batch_wait_ms=15,
    hidden=96,
    layers=3,
    cpu_threads=1,
    learning_rate=3e-4,
    clip_ratio=0.2,
    entropy_weight=0.01,
    target_kl=0.03,
    validation=(),
    validate_every=25,
    checkpoint_every=10,
    resume=None,
    dataset_metadata=None,
    progress=None,
):
    if isinstance(dispatchers, Dispatcher):
        dispatchers = [dispatchers]
    dispatchers, validation = list(dispatchers), list(validation)
    if (
        not dispatchers
        or min(
            iterations,
            max_hooks,
            groups_per_batch,
            update_epochs,
            minibatch_size,
            inference_batch_size,
            cpu_threads,
            checkpoint_every,
        )
        < 1
        or group_size < 2
    ):
        raise ValueError("positive budgets and group_size >= 2 required")
    if (
        workers < 0
        or validate_every < 0
        or learning_rate <= 0
        or not 0 < clip_ratio < 1
        or min(entropy_weight, target_kl, batch_wait_ms) < 0
        or not 0 <= branch_fraction <= 1
        or state_pool_size < 1
        or trajectory_batch_size < group_size
    ):
        raise ValueError("invalid training configuration")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or precision not in ("float32", "bfloat16"):
        raise ValueError("supported devices: cpu/cuda; precisions: float32/bfloat16")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; choose --device cpu --precision float32 explicitly"
        )
    if precision == "bfloat16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bfloat16 requires a supported CUDA GPU")
    profile = dispatchers[0].profile
    for d in dispatchers + validation:
        if d.profile != profile:
            raise ValueError(
                "all requests must use exactly the same physical/business contract"
            )
        errors = d.validate(d.env.scenario.initial, BusinessContext())
        if errors or d.done(d.env.scenario.initial, BusinessContext()):
            raise ValueError(
                "training/evaluation requires valid nontrivial requests: " + str(errors)
            )
    train_hashes = [d.env.scenario_hash for d in dispatchers]
    if set(train_hashes) & {d.env.scenario_hash for d in validation}:
        raise ValueError("training and validation requests overlap")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_file = output.with_suffix(".history.jsonl")
    config = dict(
        algorithm=ALGORITHM,
        requests=train_hashes,
        validation=[d.env.scenario_hash for d in validation],
        profile=profile,
        seed=seed,
        max_hooks=max_hooks,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        trajectory_batch_size=trajectory_batch_size,
        branch_fraction=branch_fraction,
        state_pool_size=state_pool_size,
        learning_rate=learning_rate,
        clip_ratio=clip_ratio,
        entropy_weight=entropy_weight,
        target_kl=target_kl,
        hidden=hidden,
        layers=layers,
        precision=precision,
    )
    signature = digest(config)
    model = GraphPolicy(hidden, layers, max_hooks).to(device)
    model.bind(dispatchers[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    completed = optimizer_steps = total_steps = complete_trajectories = (
        signal_groups
    ) = 0
    validation_history = []
    final_validation = None
    order = list(range(len(dispatchers)))
    rng.shuffle(order)
    cursor = 0
    pool = StatePool(state_pool_size)
    best_known = {}
    first_solution_step = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        model = load_model(resume, device)
        model.validate_profile(dispatchers[0])
        if (
            checkpoint.get("algorithm") != ALGORITHM
            or checkpoint.get("training_signature") != signature
        ):
            raise ValueError("resume dataset/training configuration mismatch")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed = checkpoint["iterations_completed"]
        optimizer_steps = checkpoint["optimizer_steps"]
        total_steps = checkpoint["environment_steps"]
        complete_trajectories = checkpoint["complete_trajectories"]
        signal_groups = checkpoint["groups_with_relative_signal"]
        validation_history = checkpoint["validation_history"]
        final_validation = checkpoint["validation_final"]
        order, cursor = checkpoint["request_order"], checkpoint["request_cursor"]
        pool.add(
            [StartPoint.from_dict(p) for p in checkpoint["state_pool"]],
            dispatchers,
            max_hooks,
        )
        best_known = checkpoint["best_known_plans"]
        first_solution_step = checkpoint["first_solution_step"]
        rng.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        # A crash can leave log rows beyond the last atomic checkpoint.
        # Keep the journal consistent with the weights being resumed.
        source_history = Path(resume).with_suffix(".history.jsonl")
        temporary_history = history_file.with_suffix(".jsonl.tmp")
        with temporary_history.open("w") as destination:
            if completed and source_history.exists():
                with source_history.open() as source:
                    for line in source:
                        row = json.loads(line)
                        if row["iteration"] > completed:
                            break
                        destination.write(line)
                        if row["iteration"] == completed:
                            # A process interruption may leave the next JSON row torn.
                            # It was never committed, so do not parse it on recovery.
                            break
        temporary_history.replace(history_file)
    runtime = {
        "device": str(device),
        "precision": precision,
        "cpu_workers": workers,
        "cpu_threads": cpu_threads,
        "inference_batch_size": inference_batch_size,
        "batch_wait_ms": batch_wait_ms,
        "minibatch_size": minibatch_size,
        "groups_per_batch": groups_per_batch,
        "trajectory_batch_size": trajectory_batch_size,
        "branch_fraction": branch_fraction,
        "group_size": group_size,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    last_timing = {}

    def save():
        checkpoint = dict(
            schema_version=CHECKPOINT_SCHEMA,
            feature_schema=FEATURE_SCHEMA,
            algorithm=ALGORITHM,
            hidden=model.hidden,
            hook_budget=max_hooks,
            state_pool=pool.export(),
            best_known_plans=best_known,
            first_solution_step=first_solution_step,
            layers=model.layers,
            profile=profile,
            state_dict=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            training_signature=signature,
            training_config=config,
            iterations_completed=completed,
            optimizer_steps=optimizer_steps,
            environment_steps=total_steps,
            complete_trajectories=complete_trajectories,
            groups_with_relative_signal=signal_groups,
            objective="feasibility_then_total_hooks",
            seed=seed,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            python_rng=rng.getstate(),
            request_order=order,
            request_cursor=cursor,
            validation_history=validation_history,
            validation_final=final_validation,
            runtime=runtime,
        )
        temp = output.with_suffix(output.suffix + ".tmp")
        torch.save(checkpoint, temp)
        temp.replace(output)
        write_json(
            output.with_suffix(".metrics.json"),
            dict(
                schema_version=CHECKPOINT_SCHEMA,
                algorithm=ALGORITHM,
                profile=profile,
                runtime=runtime,
                iterations_completed=completed,
                optimizer_steps=optimizer_steps,
                environment_steps=total_steps,
                complete_trajectories=complete_trajectories,
                groups_with_relative_signal=signal_groups,
                training_requests=len(dispatchers),
                objective="feasibility_then_total_hooks",
                dataset=dataset_metadata,
                validation_final=final_validation,
                validation_history=validation_history,
                history_file=str(history_file),
                state_pool_size=len(pool.points),
                verified_training_requests=len(best_known),
                first_solution_step=first_solution_step,
                best_known_total_hooks={k: len(v) for k, v in best_known.items()},
                last_iteration=last_timing,
                peak_gpu_memory_mb=(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0
                ),
            ),
        )

    if not resume:
        history_file.write_text("")
        save()
    target_iteration = completed + iterations
    with ParallelSampler(
        dispatchers + validation, workers, inference_batch_size, batch_wait_ms
    ) as sampler:
        while completed < target_iteration:
            requests = []
            for _ in range(groups_per_batch):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                requests.append(order[cursor])
                cursor += 1
            starts = [
                StartPoint(i, dispatchers[i].env.scenario.initial, BusinessContext())
                for i in requests
            ]
            slots = list(range(len(requests)))
            rng.shuffle(slots)
            wanted = round(len(requests) * branch_fraction)
            used = 0
            for i in slots:
                point = pool.choose(requests[i], rng)
                if point is not None and used < wanted:
                    starts[i] = point
                    used += 1
            started = time.perf_counter()
            trajectories, timing = sampler.collect(
                model,
                requests,
                group_size,
                max_hooks,
                precision=precision,
                progress=progress,
                starts=starts,
            )
            update = update_policy(
                model,
                optimizer,
                trajectories,
                group_size=group_size,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                clip_ratio=clip_ratio,
                entropy_weight=entropy_weight,
                target_kl=target_kl,
                precision=precision,
                rng=rng,
                trajectory_batch_size=trajectory_batch_size,
                progress=progress,
            )
            for group in update["groups"]:
                request = dispatchers[group["request_index"]].env.scenario
                group["request"] = request.name
                group["scenario_hash"] = dispatchers[
                    group["request_index"]
                ].env.scenario_hash
            pool.add(
                [p for t in trajectories for p in t.outcome.snapshots],
                dispatchers,
                max_hooks,
            )
            root_traces = [t for t in trajectories if not t.start_hook]
            branch_traces = [t for t in trajectories if t.start_hook]
            for t in trajectories:
                if t.outcome.complete:
                    key = dispatchers[t.request_index].env.scenario_hash
                    actions = [a.to_dict() for a in t.outcome.actions]
                    if key not in best_known or len(actions) < len(best_known[key]):
                        best_known[key] = actions
            if best_known and first_solution_step is None:
                first_solution_step = total_steps + timing["first_complete_step"]
            completed += 1
            optimizer_steps += update["optimizer_updates"]
            total_steps += timing["environment_steps"]
            complete_trajectories += sum(g["complete"] for g in update["groups"])
            signal_groups += sum(g["has_relative_signal"] for g in update["groups"])
            elapsed = time.perf_counter() - started
            last_timing = dict(
                iteration=completed,
                root_trajectories=len(root_traces),
                branch_trajectories=len(branch_traces),
                root_environment_steps=sum(
                    len(t.outcome.actions) - t.start_hook for t in root_traces
                ),
                branch_environment_steps=sum(
                    len(t.outcome.actions) - t.start_hook for t in branch_traces
                ),
                root_complete=sum(t.outcome.complete for t in root_traces),
                branch_complete=sum(t.outcome.complete for t in branch_traces),
                complete_within_40=sum(
                    t.outcome.complete and t.outcome.final_state.hook <= 40
                    for t in trajectories
                ),
                improved_from_start=sum(
                    t.outcome.score > t.outcome.start_score for t in trajectories
                ),
                stop_counts=dict(Counter(t.outcome.stop_reason for t in trajectories)),
                sum_trajectory_unique_states=sum(
                    t.outcome.unique_states for t in trajectories
                ),
                state_pool_size=len(pool.points),
                verified_training_requests=len(best_known),
                first_solution_step=first_solution_step,
                **timing,
                **update,
                iteration_seconds=elapsed,
                environment_steps_per_second=timing["environment_steps"]
                / max(elapsed, 1e-9),
                rollout_steps_per_second=timing["environment_steps"]
                / max(timing["sampling_seconds"], 1e-9),
                batch_completion_rate=sum(t.outcome.complete for t in trajectories)
                / len(trajectories),
                batch_groups_with_relative_signal=sum(
                    g["has_relative_signal"] for g in update["groups"]
                ),
                batch_mean_score=sum(t.outcome.score for t in trajectories)
                / len(trajectories),
            )
            # Release CPU decision graphs before validation or the next rollout batch.
            del trajectories, root_traces, branch_traces
            with history_file.open("a") as stream:
                stream.write(json.dumps(last_timing, ensure_ascii=False) + "\n")
            if progress:
                progress(
                    {
                        k: v
                        for k, v in last_timing.items()
                        if k not in ("groups", "update_epochs")
                    }
                )
            if validation and (
                completed == target_iteration
                or validate_every
                and completed % validate_every == 0
            ):
                final_validation = evaluate(
                    model,
                    validation,
                    max_hooks,
                    sampler=sampler,
                    request_indices=list(
                        range(len(dispatchers), len(dispatchers) + len(validation))
                    ),
                    precision=precision,
                )
                validation_history.append(
                    {
                        "iteration": completed,
                        **{k: v for k, v in final_validation.items() if k != "results"},
                    }
                )
            if (
                completed == target_iteration
                or completed == 1
                or completed % checkpoint_every == 0
            ):
                save()
    return {
        "checkpoint": str(output),
        "iterations": completed,
        "updates": optimizer_steps,
        "environment_steps": total_steps,
        "complete_trajectories": complete_trajectories,
        "groups_with_relative_signal": signal_groups,
        "training_requests": len(dispatchers),
        "runtime": runtime,
        "verified_training_requests": len(best_known),
        "first_solution_step": first_solution_step,
        "last_iteration": {
            k: v for k, v in last_timing.items() if k not in ("groups", "update_epochs")
        },
        "validation": final_validation,
    }
