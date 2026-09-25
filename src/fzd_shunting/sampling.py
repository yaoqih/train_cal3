"""CPU environment actors with central batched policy inference.

The policy stays frozen for an entire collected batch. Actors never own a GPU
or a model and never receive stale actions from a previous policy update.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import torch

from .dispatch import Dispatcher
from .domain import Action, BusinessContext, State, digest
from .environment import Environment
from .objective import trajectory_score
from .policy.gnn import collate_graphs
from .decisions import DecisionCache
from collections import OrderedDict


@dataclass
class ScenarioSpec:
    yard: object
    scenario: object
    rules: object
    constraints: tuple
    profile: dict

    @classmethod
    def from_dispatcher(cls, dispatcher):
        return cls(
            dispatcher.env.yard,
            dispatcher.env.scenario,
            dispatcher.rules,
            dispatcher.constraints,
            dispatcher.profile,
        )

    def build(self):
        d = Dispatcher(
            Environment(self.yard, self.scenario), self.rules, self.constraints
        )
        if d.profile != self.profile:
            raise ValueError(
                "actor environment does not match the requested execution contract"
            )
        return d


@dataclass(frozen=True)
class StartPoint:
    request_index: int
    state: object
    context: object
    prefix: tuple = ()

    def to_dict(self):
        return dict(
            request_index=self.request_index,
            state=self.state.to_dict(),
            context=self.context.to_dict(),
            prefix=[a.to_dict() for a in self.prefix],
        )

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["request_index"],
            State.from_dict(value["state"]),
            BusinessContext.from_dict(value["context"]),
            tuple(Action(**a) for a in value["prefix"]),
        )


class StatePool:
    """Only real training prefixes; repeated equivalent layouts retain their shortest prefix."""

    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.points = OrderedDict()
        self.by_request = {}

    def add(self, points, dispatchers, hook_budget):
        for point in points:
            if not 0 <= point.request_index < len(dispatchers):
                raise ValueError("validation/test state must not enter training pool")
            d = dispatchers[point.request_index]
            if not 0 < point.state.hook <= hook_budget - 4:
                continue
            key = (point.request_index, d.search_key(point.state, point.context))
            old = self.points.get(key)
            if old is not None and old.state.hook <= point.state.hook:
                continue
            self.points[key] = point
            self.points.move_to_end(key)
            bucket = self.by_request.setdefault(point.request_index, OrderedDict())
            bucket[key] = None
            bucket.move_to_end(key)
            # Keep states across a full dataset pass; a global FIFO would retain
            # only the most recently sampled requests and starve later forks.
            quota = max(1, self.capacity // len(dispatchers))
            while len(bucket) > quota:
                removed, _ = bucket.popitem(last=False)
                self.points.pop(removed)
            if len(self.points) > self.capacity:
                removed, _ = self.points.popitem(last=False)
                self.by_request[removed[0]].pop(removed)

    def choose(self, request_index, rng):
        options = [p for p in self.points.values() if p.request_index == request_index]
        if not options:
            return None
        # Sample hook-depth buckets evenly, without preferring already-good goal scores.
        buckets = {}
        for p in options:
            buckets.setdefault(p.state.hook // 8, []).append(p)
        return rng.choice(buckets[rng.choice(sorted(buckets))])

    def export(self):
        return [p.to_dict() for p in self.points.values()]


@dataclass
class Outcome:
    final_state: object
    final_context: object
    complete: bool
    score: float
    stop_reason: str
    actions: tuple
    snapshots: tuple = ()
    unique_states: int = 0
    start_score: float = 0.0


@dataclass
class Sample:
    graph: object
    action_index: int
    old_log_probs: np.ndarray


@dataclass
class Trajectory:
    request_index: int
    group_index: int
    start_key: str = ""
    start_hook: int = 0
    samples: list = field(default_factory=list)
    outcome: object = None


class Actor:
    def __init__(self, specs):
        self.dispatchers = [spec.build() for spec in specs]
        self.lanes = {}
        self.cache = DecisionCache()
        self.prefixes = OrderedDict()

    def reset(self, jobs, max_hooks, record):
        self.max_hooks, self.record = max_hooks, record
        self.cache_base = self.cache.hits
        self.lanes = {}
        for lane, point in jobs:
            d = self.dispatchers[point.request_index]
            key = (point.request_index, point.prefix)
            if key not in self.prefixes:
                state, ctx = d.env.scenario.initial, BusinessContext()
                seen = set()
                for action in point.prefix:
                    seen.add(d.search_key(state, ctx))
                    c = d.prepare(state, action, ctx)
                    state, ctx = c.after, c.after_context
                if state != point.state or ctx != point.context:
                    raise ValueError(
                        "branch state does not match original request and legal prefix"
                    )
                self.prefixes[key] = (frozenset(seen), state, ctx)
                if len(self.prefixes) > 256:
                    self.prefixes.popitem(last=False)
            seen, expected_state, expected_context = self.prefixes[key]
            if point.state != expected_state or point.context != expected_context:
                raise ValueError(
                    "branch state does not match original request and legal prefix"
                )
            if point.state.hook >= max_hooks or d.done(point.state, point.context):
                raise ValueError(
                    "rollout start must be nonterminal and within global budget"
                )
            self.lanes[lane] = dict(
                d=d,
                request=point.request_index,
                state=point.state,
                context=point.context,
                seen=set(seen),
                actions=list(point.prefix),
                snapshots=[],
                start_score=trajectory_score(d, point.state, point.context, max_hooks),
            )
        return self.views()

    def advance(self, choices):
        if set(choices) != set(self.lanes):
            raise ValueError("actor action batch does not match active lanes")
        for lane, index in choices.items():
            slot = self.lanes[lane]
            c = slot.pop("candidates")[index]
            slot["state"], slot["context"] = c.after, c.after_context
            slot["actions"].append(c.action)
        return self.views()

    def views(self):
        ready, outcomes = [], []
        for lane, slot in list(self.lanes.items()):
            d, state, ctx = slot["d"], slot["state"], slot["context"]
            complete = d.done(state, ctx)
            key = d.search_key(state, ctx)
            stop = (
                "complete"
                if complete
                else (
                    "hook_budget"
                    if state.hook >= self.max_hooks
                    else "cycle" if key in slot["seen"] else None
                )
            )
            if stop is None:
                slot["seen"].add(key)
                candidates, graph, identity = self.cache.get(
                    d, state, ctx, self.max_hooks
                )
                if not candidates:
                    stop = "dead_end"
            if stop:
                outcomes.append(
                    (
                        lane,
                        Outcome(
                            state,
                            ctx,
                            complete,
                            trajectory_score(
                                d,
                                state,
                                ctx,
                                self.max_hooks,
                                stop if not complete else None,
                            ),
                            stop,
                            tuple(slot["actions"]),
                            tuple(slot["snapshots"]),
                            len(slot["seen"]),
                            slot["start_score"],
                        ),
                    )
                )
                del self.lanes[lane]
            else:
                if self.record and state.hook and state.hook % 4 == 0:
                    slot["snapshots"].append(
                        StartPoint(slot["request"], state, ctx, tuple(slot["actions"]))
                    )
                slot["candidates"] = candidates
                ready.append((lane, identity, graph))
        return ready, outcomes, self.cache.hits - self.cache_base


def _actor_process(connection, specs):
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        actor = Actor(specs)
        connection.send(("started", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            result = (
                actor.reset(*payload) if command == "reset" else actor.advance(payload)
            )
            connection.send(("views", result))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class ParallelSampler:
    def __init__(
        self, dispatchers, workers=8, inference_batch_size=64, batch_wait_ms=15
    ):
        if workers < 0 or inference_batch_size < 1 or batch_wait_ms < 0:
            raise ValueError("invalid actor or inference batch settings")
        self.specs = [ScenarioSpec.from_dispatcher(d) for d in dispatchers]
        self.workers = workers
        self.inference_batch_size = inference_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.processes, self.connections = [], []
        self.local = Actor(self.specs) if not workers else None
        try:
            context = mp.get_context("spawn")
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_actor_process, args=(child, self.specs), daemon=True
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            for connection in self.connections:
                if not connection.poll(60):
                    raise RuntimeError("CPU actor startup timed out")
                tag, payload = connection.recv()
                if tag != "started":
                    raise RuntimeError("CPU actor startup failed: " + str(payload))
        except BaseException:
            self.close()
            raise

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def collect(
        self,
        model,
        request_indices,
        group_size,
        max_hooks,
        *,
        precision="float32",
        greedy=False,
        progress=None,
        starts=None,
        record=True,
    ):
        record = record and not greedy
        device = next(model.parameters()).device
        if group_size < 1 or max_hooks < 1:
            raise ValueError("positive group size and hook budget required")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("supported precisions: float32/bfloat16")
        if precision == "bfloat16" and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 sampling requires a supported CUDA GPU")
        for spec in self.specs:
            if model.profile != spec.profile:
                raise ValueError("sampling policy contract mismatch")
        model.eval()
        if starts is None:
            starts = [
                StartPoint(i, self.specs[i].scenario.initial, BusinessContext())
                for i in request_indices
            ]
        if len(starts) != len(request_indices) or any(
            p.request_index != i for p, i in zip(starts, request_indices)
        ):
            raise ValueError("start points do not match request groups")
        trajectories = [
            Trajectory(p.request_index, group, digest(p.to_dict()), p.state.hook)
            for group, p in enumerate(starts)
            for _ in range(group_size)
        ]
        jobs = [(i, starts[t.group_index]) for i, t in enumerate(trajectories)]
        started = time.perf_counter()
        inference_seconds = packing_seconds = 0.0
        inference_batches = inference_samples = steps = inference_hits = 0
        first_complete_step = None
        distinct_decisions = set()
        actor_hits = {}
        last_progress = started
        inference_cache = OrderedDict()

        def decide(responses):
            nonlocal inference_seconds, packing_seconds, inference_batches, inference_samples, steps, inference_hits
            nonlocal first_complete_step
            views, missing = [], {}
            choices = {worker: {} for worker, _ in responses}
            for worker, (ready, outcomes, hits) in responses:
                actor_hits[worker] = hits
                for lane, outcome in outcomes:
                    trajectories[lane].outcome = outcome
                    if outcome.complete and first_complete_step is None:
                        first_complete_step = steps
                for lane, identity, graph in ready:
                    distinct_decisions.add(identity)
                    views.append((worker, lane, identity, graph))
                    if identity not in inference_cache and identity not in missing:
                        missing[identity] = graph
                    else:
                        inference_hits += 1
            items = list(missing.items())
            for first in range(0, len(items), self.inference_batch_size):
                part = items[first : first + self.inference_batch_size]
                t = time.perf_counter()
                batch = collate_graphs([g for _, g in part])
                if device.type == "cuda":
                    batch = batch.pin_memory()
                batch = batch.to(device)
                packing_seconds += time.perf_counter() - t
                t = time.perf_counter()
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=precision == "bfloat16",
                    ),
                ):
                    probabilities = (
                        torch.log_softmax(model.forward_batch(batch).float(), dim=1)
                        .cpu()
                        .numpy()
                    )
                inference_seconds += time.perf_counter() - t
                inference_batches += 1
                inference_samples += len(part)
                for j, (identity, graph) in enumerate(part):
                    inference_cache[identity] = probabilities[
                        j, : len(graph.action_features)
                    ].copy()
            if views:
                width = max(len(inference_cache[key]) for _, _, key, _ in views)
                probs = np.zeros((len(views), width), dtype=np.float32)
                for j, (_, _, key, _) in enumerate(views):
                    lp = inference_cache[key]
                    probs[j, : len(lp)] = np.exp(lp)
                values = torch.from_numpy(probs)
                selected = (
                    values.argmax(1)
                    if greedy
                    else torch.multinomial(values, 1).squeeze(1)
                ).tolist()
                for j, (worker, lane, identity, graph) in enumerate(views):
                    choices[worker][lane] = selected[j]
                    trajectory = trajectories[lane]
                    # Forks credit only the first new decision; later graphs are
                    # needed for inference, but need not survive until the update.
                    if record and (not trajectory.start_hook or not trajectory.samples):
                        trajectory.samples.append(
                            Sample(graph, selected[j], inference_cache[identity])
                        )
                    steps += 1
            while len(inference_cache) > 4096:
                inference_cache.popitem(last=False)
            return choices

        if self.local is not None:
            response = self.local.reset(jobs, max_hooks, record)
            while True:
                choices = decide([(0, response)])[0]
                if not choices:
                    break
                response = self.local.advance(choices)
                if progress and time.perf_counter() - last_progress >= 15:
                    progress(
                        {
                            "phase": "sampling",
                            "environment_steps": steps,
                            "seconds": round(time.perf_counter() - started, 2),
                        }
                    )
                    last_progress = time.perf_counter()
        else:
            active = set()
            for worker, connection in enumerate(self.connections):
                subset = [
                    job
                    for job in jobs
                    if trajectories[job[0]].group_index % self.workers == worker
                ]
                if subset:
                    connection.send(("reset", (subset, max_hooks, record)))
                    active.add(worker)
            while active:
                pending = [self.connections[i] for i in sorted(active)]
                ready = wait(pending, timeout=10)
                if not ready:
                    if any(not self.processes[i].is_alive() for i in active):
                        raise RuntimeError("CPU actor exited while sampling")
                    continue
                responses = []
                deadline = time.perf_counter() + self.batch_wait
                remaining = set(active)
                while ready:
                    for connection in ready:
                        worker = self.connections.index(connection)
                        remaining.remove(worker)
                        tag, result = connection.recv()
                        if tag != "views":
                            raise RuntimeError("CPU actor failed: " + str(result))
                        responses.append((worker, result))
                    if (
                        not remaining
                        or time.perf_counter() >= deadline
                        or sum(len(x[1][0]) for x in responses)
                        >= self.inference_batch_size
                    ):
                        break
                    ready = wait(
                        [self.connections[i] for i in sorted(remaining)],
                        timeout=max(0, deadline - time.perf_counter()),
                    )
                choices = decide(responses)
                for worker, actions in choices.items():
                    if actions:
                        self.connections[worker].send(("step", actions))
                    else:
                        active.remove(worker)
                if progress and time.perf_counter() - last_progress >= 15:
                    progress(
                        {
                            "phase": "sampling",
                            "environment_steps": steps,
                            "seconds": round(time.perf_counter() - started, 2),
                        }
                    )
                    last_progress = time.perf_counter()
        if any(t.outcome is None for t in trajectories):
            raise RuntimeError("incomplete actor response")
        return trajectories, {
            "sampling_seconds": time.perf_counter() - started,
            "inference_seconds": inference_seconds,
            "packing_seconds": packing_seconds,
            "inference_batches": inference_batches,
            "environment_steps": steps,
            "mean_inference_batch": inference_samples / max(1, inference_batches),
            "inference_samples": inference_samples,
            "inference_cache_hits": inference_hits,
            "decision_cache_hits": sum(actor_hits.values()),
            "distinct_decision_states": len(distinct_decisions),
            "first_complete_step": first_complete_step,
        }
