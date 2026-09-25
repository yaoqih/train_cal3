from __future__ import annotations

import heapq
import itertools
import json
import math
import time
from dataclasses import dataclass

from .domain import Action, BusinessContext, InvalidAction, State
from .policy.rule import RulePolicy


def replay(dispatcher, actions, events=None, *, start=None, context=None):
    if events is not None and len(events) != len(actions):
        raise InvalidAction("REPLAY_EVENT_COUNT")
    state = dispatcher.reset(start, context)
    for i, item in enumerate(actions):
        action = item if isinstance(item, Action) else Action(**item)
        state = dispatcher.step(state, dispatcher.prepare(state, action))
        if events is not None:
            expected, actual = events[i], dispatcher.events[-1]
            if expected.get("schema_version") != 2:
                raise InvalidAction("REPLAY_SCHEMA: expected train_cal3 event schema 2")
            if json.dumps(actual, sort_keys=True) != json.dumps(
                expected, sort_keys=True
            ):
                raise InvalidAction("REPLAY_MISMATCH: event " + str(i + 1))
    return state


@dataclass
class SearchNode:
    state: State
    context: BusinessContext
    parent: object = None
    action: object = None
    log_cost: float = 0.0


def _actions(node):
    result = []
    while node.parent is not None:
        result.append(node.action)
        node = node.parent
    return tuple(reversed(result))


def _result(dispatcher, actions, start, context, status, **metadata):
    final = replay(dispatcher, actions, start=start, context=context)
    done = dispatcher.done(final)
    return {
        "schema_version": 2,
        "status": "complete" if done else status,
        "hooks": len(actions),
        "total_hooks": final.hook,
        "actions": [a.to_dict() if isinstance(a, Action) else a for a in actions],
        "events": dispatcher.events,
        "initial_snapshot": {
            "state": start.to_dict(),
            "business_context": context.to_dict(),
            "contract_hash": dispatcher.contract_hash,
            "scenario_hash": dispatcher.env.scenario_hash,
        },
        "final_snapshot": dispatcher.snapshot(),
        "goal_errors": dispatcher.goal_errors(final),
        "completed_vehicles": dispatcher.progress(final),
        "total_vehicles": len(dispatcher.env.cars),
        "profile": dispatcher.profile,
        "route_model": dispatcher.env.route_model,
        "model_scope": dispatcher.env.capabilities,
        **metadata,
    }


def plan(
    dispatcher,
    max_expansions=200,
    max_hooks=64,
    branch_limit=40,
    *,
    policy=None,
    time_limit=180.0,
    max_frontier=4000,
    max_states=50000,
    start=None,
    context=None,
    anytime=True,
    inference_batch_size=32,
    workers=0,
    precision="float32",
):
    """Budgeted policy-guided search. A saved complete plan is improved until budget ends.

    Branch/frontier pruning is approximate; no optimality/infeasibility claim is made.
    All branches are pure simulations, with one final journal replay of the selected plan.
    """
    if (
        min(
            max_expansions,
            max_hooks,
            branch_limit,
            max_frontier,
            max_states,
            inference_batch_size,
        )
        < 1
        or time_limit <= 0
    ):
        raise ValueError("search budgets must be positive")
    initial = dispatcher.env.scenario.initial if start is None else start
    context = dispatcher.resolve_context(initial, context)
    errors = dispatcher.validate(initial, context)
    if errors:
        return {
            "schema_version": 2,
            "status": "invalid_request",
            "errors": errors,
            "actions": [],
            "events": [],
        }
    if initial.hook > max_hooks:
        raise ValueError("max_hooks is a total-plan ceiling, below current hook")
    policy = policy or RulePolicy()
    if hasattr(policy, "validate_profile"):
        policy.validate_profile(dispatcher)
    started = time.monotonic()
    root = SearchNode(initial, context)
    serial = itertools.count()
    queue = [(0.0, next(serial), root)]
    visited = {dispatcher.search_key(initial, context): initial.hook}
    best, incumbent = root, root if dispatcher.done(initial, context) else None
    best_score = (dispatcher.quality(initial), -initial.hook)
    expansions = pruned = 0
    curve = []
    stop = "frontier_exhausted"
    if incumbent:
        queue.clear()
    evaluator = None
    if hasattr(policy, "forward_batch"):
        from .frontier import FrontierEvaluator

        evaluator = FrontierEvaluator(dispatcher, policy, max_hooks, workers, precision)
    failed = True
    try:
        while queue:
            if time.monotonic() - started >= time_limit:
                stop = "time_limit"
                break
            if expansions >= max_expansions:
                stop = "expansion_limit"
                break
            batch_nodes = []
            while queue and len(batch_nodes) < min(
                inference_batch_size, max_expansions - expansions
            ):
                _, _, node = heapq.heappop(queue)
                state, ctx = node.state, node.context
                if (
                    visited.get(dispatcher.search_key(state, ctx), state.hook)
                    < state.hook
                ):
                    continue
                if state.hook >= max_hooks or (
                    incumbent is not None and state.hook >= incumbent.state.hook - 1
                ):
                    continue
                batch_nodes.append(node)
            if not batch_nodes:
                continue
            if evaluator is not None:
                batches = evaluator.rank(batch_nodes)
            else:
                batches = []
                for node in batch_nodes:
                    candidates = dispatcher.candidates(node.state, node.context)
                    batches.append(
                        [
                            (c, -i / max(1, len(candidates)))
                            for i, c in enumerate(policy.rank(dispatcher, candidates))
                        ]
                    )
            for node, ranked in zip(batch_nodes, batches):
                if time.monotonic() - started >= time_limit:
                    stop = "time_limit"
                    break
                state, ctx = node.state, node.context
                if (
                    visited.get(dispatcher.search_key(state, ctx), state.hook)
                    < state.hook
                ):
                    continue
                if incumbent is not None and state.hook >= incumbent.state.hook - 1:
                    continue
                expansions += 1
                # Preserve line/operation diversity instead of spending all slots on one prefix.
                primary = ranked[: max(1, branch_limit * 3 // 4)]
                selected = {c.action for c, _ in primary}
                kinds = {(c.action.line, c.action.operation) for c, _ in primary}
                for item in ranked:
                    c, _ = item
                    if len(primary) >= branch_limit:
                        break
                    if (c.action.line, c.action.operation) not in kinds:
                        primary.append(item)
                        selected.add(c.action)
                        kinds.add((c.action.line, c.action.operation))
                for item in ranked:
                    if len(primary) >= branch_limit:
                        break
                    if item[0].action not in selected:
                        primary.append(item)
                        selected.add(item[0].action)
                pruned += max(0, len(ranked) - len(primary))
                for candidate, logp in primary:
                    nxt, next_ctx = candidate.after, candidate.after_context
                    key = dispatcher.search_key(nxt, next_ctx)
                    if visited.get(key, math.inf) <= nxt.hook:
                        continue
                    if incumbent is not None and nxt.hook >= incumbent.state.hook:
                        continue
                    if len(visited) >= max_states:
                        stop = "state_limit"
                        break
                    visited[key] = nxt.hook
                    child = SearchNode(
                        nxt, next_ctx, node, candidate.action, node.log_cost - logp
                    )
                    score = (dispatcher.quality(nxt), -nxt.hook)
                    if score > best_score:
                        best, best_score = child, score
                    if dispatcher.done(nxt, next_ctx):
                        incumbent = child
                        curve.append(
                            {
                                "seconds": round(time.monotonic() - started, 6),
                                "total_hooks": nxt.hook,
                            }
                        )
                        if not anytime:
                            stop = "first_solution"
                            break
                        continue
                    remaining = 1.75 * len(dispatcher.env.cars) - dispatcher.quality(
                        nxt
                    )
                    priority = nxt.hook + 2 * remaining + 0.08 * child.log_cost
                    heapq.heappush(queue, (priority, next(serial), child))
                if stop in ("state_limit", "first_solution"):
                    break
            if stop in ("time_limit", "state_limit", "first_solution"):
                break
            if len(queue) > max_frontier:
                pruned += len(queue) - max_frontier
                queue = heapq.nsmallest(max_frontier, queue)
                heapq.heapify(queue)
        failed = False
    finally:
        if evaluator is not None:
            evaluator.close(failed)
    elapsed = time.monotonic() - started
    chosen = incumbent or best
    actions = _actions(chosen)
    return _result(
        dispatcher,
        actions,
        initial,
        context,
        "search_limit",
        expanded_states=expansions,
        visited_states=len(visited),
        pruned=pruned,
        stop_reason=stop,
        elapsed_seconds=elapsed,
        improvement_curve=curve,
        inference=evaluator.stats if evaluator is not None else {},
        search_limits={
            "expansions": max_expansions,
            "hooks": max_hooks,
            "branch": branch_limit,
            "seconds": time_limit,
            "frontier": max_frontier,
            "states": max_states,
            "inference_batch_size": inference_batch_size,
            "workers": workers,
            "precision": precision,
        },
        note="Budgeted approximate search; no optimality or infeasibility certificate.",
    )


def rollout(dispatcher, policy, max_hooks=64, *, start=None, context=None):
    initial = dispatcher.env.scenario.initial if start is None else start
    if hasattr(policy, "hook_budget"):
        policy.hook_budget = max_hooks
    context = dispatcher.resolve_context(initial, context)
    state = dispatcher.reset(initial, context)
    if max_hooks < state.hook:
        raise ValueError("max_hooks is below current hook")
    if hasattr(policy, "validate_profile"):
        policy.validate_profile(dispatcher)
    visited, actions = set(), []
    status, reasons = "search_limit", {}
    while state.hook < max_hooks and not dispatcher.done(state):
        key = dispatcher.search_key(state, dispatcher.context)
        if key in visited:
            status = "cycle_detected"
            break
        visited.add(key)
        candidates = dispatcher.candidates(state, diagnostics=reasons)
        if not candidates:
            status = "dead_end"
            break
        selected = policy.choose(dispatcher, state, candidates, dispatcher.context)
        state = dispatcher.step(state, selected)
        actions.append(selected.action)
    return _result(
        dispatcher, actions, initial, context, status, rejection_reasons=reasons
    )


def next_hook(dispatcher, policy, max_hooks=64):
    """One decision for the supplied observed state; no commit or simulated observation."""
    if max_hooks < 1 or max_hooks < dispatcher.state.hook:
        raise ValueError("max_hooks must be positive and not below current hook")
    if hasattr(policy, "hook_budget"):
        policy.hook_budget = max_hooks
    if hasattr(policy, "validate_profile"):
        policy.validate_profile(dispatcher)
    state, context = dispatcher.state, dispatcher.context
    if dispatcher.done(state, context):
        return {
            "schema_version": 2,
            "status": "complete",
            "action": None,
            "current_snapshot": dispatcher.snapshot(),
        }
    if state.hook >= max_hooks:
        return {
            "schema_version": 2,
            "status": "hook_budget",
            "action": None,
            "current_snapshot": dispatcher.snapshot(),
        }
    reasons = {}
    candidates = dispatcher.candidates(state, context, diagnostics=reasons)
    if not candidates:
        return {
            "schema_version": 2,
            "status": "dead_end",
            "action": None,
            "rejection_reasons": reasons,
            "current_snapshot": dispatcher.snapshot(),
        }
    selected = policy.choose(dispatcher, state, candidates, context)
    return {
        "schema_version": 2,
        "status": "action",
        "action": selected.action.to_dict(),
        "vehicles": selected.vehicles,
        "route": selected.route,
        "expected_decision": selected.before_key,
        "current_snapshot": dispatcher.snapshot(),
        "predicted_snapshot": {
            "state": selected.after.to_dict(),
            "business_context": selected.after_context.to_dict(),
            "contract_hash": dispatcher.contract_hash,
            "scenario_hash": dispatcher.env.scenario_hash,
        },
    }
