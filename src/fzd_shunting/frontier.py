"""CPU frontier preparation and frozen-policy GPU batches for bounded search."""

from __future__ import annotations
import multiprocessing as mp
from collections import OrderedDict
import time
import torch
from .sampling import ScenarioSpec
from .decisions import DecisionCache
from .policy.gnn import collate_graphs


def _initialize(spec):
    global _dispatcher, _cache
    torch.set_num_threads(1)
    _dispatcher, _cache = spec.build(), DecisionCache()


def _prepare(job):
    state, context, budget = job
    return _cache.get(_dispatcher, state, context, budget)


class FrontierEvaluator:
    def __init__(self, dispatcher, policy, hook_budget, workers=0, precision="float32"):
        self.dispatcher, self.policy, self.budget = dispatcher, policy, hook_budget
        policy.validate_profile(dispatcher)
        device = next(policy.parameters()).device
        if precision == "bfloat16" and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 search requires a supported CUDA GPU")
        self.precision, self.workers = precision, workers
        if workers < 0 or precision not in ("float32", "bfloat16"):
            raise ValueError("invalid search runtime")
        self.cache = DecisionCache()
        self.probabilities = OrderedDict()
        self.pool = None
        self.stats = dict(
            inference_batches=0,
            inference_states=0,
            inference_seconds=0.0,
            preparation_seconds=0.0,
            inference_cache_hits=0,
        )
        if workers:
            self.pool = mp.get_context("spawn").Pool(
                workers,
                initializer=_initialize,
                initargs=(ScenarioSpec.from_dispatcher(dispatcher),),
            )

    def close(self, failed=False):
        if self.pool is not None:
            self.pool.terminate() if failed else self.pool.close()
            self.pool.join()
            self.pool = None

    def rank(self, nodes):
        started = time.perf_counter()
        jobs = [(n.state, n.context, self.budget) for n in nodes]
        prepared = (
            self.pool.map(_prepare, jobs)
            if self.pool
            else [self.cache.get(self.dispatcher, *j) for j in jobs]
        )
        self.stats["preparation_seconds"] += time.perf_counter() - started
        missing = {}
        for candidates, graph, key in prepared:
            if candidates:
                if key in self.probabilities or key in missing:
                    self.stats["inference_cache_hits"] += 1
                else:
                    missing[key] = graph
        if missing:
            items = list(missing.items())
            device = next(self.policy.parameters()).device
            batch = collate_graphs([g for _, g in items])
            if device.type == "cuda":
                batch = batch.pin_memory()
            batch = batch.to(device)
            started = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=self.precision == "bfloat16",
                ),
            ):
                scores = (
                    torch.log_softmax(self.policy.forward_batch(batch).float(), dim=1)
                    .cpu()
                    .tolist()
                )
            self.stats["inference_seconds"] += time.perf_counter() - started
            self.stats["inference_batches"] += 1
            self.stats["inference_states"] += len(items)
            for (key, graph), row in zip(items, scores):
                self.probabilities[key] = row[: len(graph.action_features)]
        result = [
            sorted(zip(cs, self.probabilities[key]), key=lambda x: -x[1]) if cs else []
            for cs, _, key in prepared
        ]
        while len(self.probabilities) > 4096:
            self.probabilities.popitem(last=False)
        return result
