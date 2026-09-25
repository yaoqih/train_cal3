"""Bounded exact-state reuse, shared by rollout and frontier expansion."""

from collections import OrderedDict
from .policy.gnn import encode_graph


class DecisionCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, dispatcher, state, context, hook_budget):
        key = (
            dispatcher.env.scenario_hash,
            dispatcher.contract_hash,
            state,
            context,
            hook_budget,
        )
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        candidates = dispatcher.candidates(state, context)
        graph = encode_graph(dispatcher, state, candidates, context, hook_budget)
        identity = (
            (
                candidates[0].before_key
                if candidates
                else dispatcher.decision_key(state, context)
            )
            + ":"
            + str(hook_budget)
        )
        value = candidates, graph, identity
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value
