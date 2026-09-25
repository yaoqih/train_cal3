from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace

from .constraints import DEFAULT_CONSTRAINTS
from .domain import Action, BusinessContext, InvalidAction, Rules, State, digest
from .environment import Environment
from .goals import GOAL_SEMANTICS, GoalEvaluator


@dataclass(frozen=True)
class Candidate:
    action: Action
    before_key: str
    after: State
    vehicles: tuple[str, ...]
    route: tuple[str, ...]
    gain: int
    destination_count: int
    context: BusinessContext
    after_context: BusinessContext
    delivered_gain: int = 0
    suffix_gain: int = 0


class Dispatcher:
    """Pure candidate simulation plus a separately versioned committed session."""

    def __init__(self, env: Environment, rules=Rules(), constraints=None):
        self.env = env
        self.rules = rules
        self.constraints = tuple(
            DEFAULT_CONSTRAINTS if constraints is None else constraints
        )
        self.protected = env.scenario.protected
        self.protected_lines = frozenset(
            env.cars[no].initial_line for no in self.protected
        )
        self.inner_capacities = dict(env.scenario.inner_capacities)
        self.goals = GoalEvaluator(env)
        self.order_assignment = self.goals.assignment
        self.rules_hash = digest(
            {
                "rules": asdict(rules),
                "constraints": [
                    {"id": c.rule_id, "version": c.version} for c in self.constraints
                ],
            }
        )
        self.profile = {
            "yard_hash": env.yard_hash,
            "rules_hash": self.rules_hash,
            "route_model": env.route_model,
            "goal_semantics": GOAL_SEMANTICS,
            "execution_semantics": "single-track-hook-context-v2",
        }
        self.contract_hash = digest(self.profile)
        self.events = []
        self.state = env.scenario.initial
        self.context = BusinessContext()
        self._locked_suffixes = {}
        for line in self.protected_lines:
            ids = env.scenario.initial.stack(line)
            i = min(i for i, no in enumerate(ids) if no in self.protected)
            self._locked_suffixes[line] = ids[i:]
        for rule in (*rules.gates, *rules.action_limits):
            if not rule.lines or not set(rule.lines) <= env.tracks.keys():
                raise ValueError("business rule contains unknown or empty tracks")
        self.max_gate = max([1] + [g.earliest_hook for g in rules.gates])

    def resolve_context(self, state, context=None):
        if context is not None:
            return context
        if state == self.state:
            return self.context
        if self.rules.action_limits and state.hook:
            raise ValueError(
                "current business_context is required for history-dependent rules"
            )
        return BusinessContext()

    def validate(self, state, context=None):
        errors = self.env.validate_state(state)
        if errors:
            return errors
        context = self.resolve_context(state, context)
        known = {q.name: q for q in self.rules.action_limits}
        counts = dict(context.counters)
        if len(counts) != len(context.counters) or set(counts) - known.keys():
            errors.append("BUSINESS_CONTEXT_KEYS")
        for key, count in context.counters:
            if type(count) is not int or count < 0 or count > state.hook:
                errors.append("BUSINESS_CONTEXT_COUNT: " + key)
            elif key in known and count > known[key].max_actions:
                errors.append("ACTION_LIMIT: " + key)
        for line, suffix in self._locked_suffixes.items():
            if state.stack(line)[-len(suffix) :] != suffix:
                errors.append("PROTECTED_ANCHOR_OR_SUFFIX: " + line)
        return errors

    def reset(self, state=None, context=None):
        state = self.env.scenario.initial if state is None else state
        if context is None and self.rules.action_limits and state.hook:
            raise ValueError("business_context required when resuming")
        context = context or BusinessContext()
        errors = self.validate(state, context)
        if errors:
            raise ValueError("invalid decision state: " + "; ".join(errors))
        self.state, self.context, self.events = state, context, []
        return state

    def decision_key(self, state, context):
        return digest(
            {
                "state": state.fingerprint(),
                "context": context.to_dict(),
                "contract": self.contract_hash,
                "scenario": self.env.scenario_hash,
            }
        )

    def search_key(self, state, context):
        return replace(state, hook=min(state.hook, self.max_gate - 1)), context

    def goal_errors(self, state, context=None):
        context = self.resolve_context(state, context)
        errors = self.validate(state, context)
        if errors:
            return errors
        errors = self.goals.errors(state)
        for constraint in self.constraints:
            errors.extend(constraint.terminal(self, state, context))
        return errors

    def done(self, state, context=None):
        return not self.goal_errors(state, context)

    def progress(self, state):
        return sum(
            len(ids)
            for line, ids in state.stacks
            if self.order_assignment(ids, line) is not None
        )

    def quality(self, state):
        metrics = self.goals.metrics(state)
        return (
            metrics["complete_lines_vehicles"]
            + 0.5 * metrics["ordered_suffix"]
            + 0.25 * metrics["delivered"]
        )

    def remaining_quota(self, line, operation, context):
        quotas = [
            max(0, q.max_actions - context.count(q.name))
            for q in self.rules.action_limits
            if line in q.lines and operation in q.operations
        ]
        return min(quotas) if quotas else None

    def _prepare(self, state, context, action, before_key):
        for constraint in self.constraints:
            constraint.before(self, state, context, action)
        after, block, route = self.env.transition(state, action, validated=True)
        for constraint in self.constraints:
            constraint.after(self, state, context, action, after, block)
        next_context = context
        for constraint in self.constraints:
            next_context = constraint.advance(self, next_context, action)
        old_ids, new_ids = state.stack(action.line), after.stack(action.line)
        old_progress = (
            len(old_ids)
            if self.order_assignment(old_ids, action.line) is not None
            else 0
        )
        new_progress = (
            len(new_ids)
            if self.order_assignment(new_ids, action.line) is not None
            else 0
        )
        delivered0, suffix0 = self.goals.line_metrics(old_ids, action.line)
        delivered1, suffix1 = self.goals.line_metrics(new_ids, action.line)
        target_count = sum(
            self.env.cars[no].target(action.line) is not None for no in block
        )
        return Candidate(
            action,
            before_key,
            after,
            block,
            route,
            new_progress - old_progress,
            target_count,
            context,
            next_context,
            delivered1 - delivered0,
            suffix1 - suffix0,
        )

    def prepare(self, state, action, context=None):
        context = self.resolve_context(state, context)
        errors = self.validate(state, context)
        if errors:
            raise InvalidAction("INVALID_STATE: " + "; ".join(errors))
        return self._prepare(state, context, action, self.decision_key(state, context))

    def candidates(self, state, context=None, *, diagnostics=None):
        context = self.resolve_context(state, context)
        errors = self.validate(state, context)
        if errors:
            raise ValueError("invalid state: " + "; ".join(errors))
        result = []
        before_key = self.decision_key(state, context)
        for line, track in self.env.tracks.items():
            if not track.parking:
                continue
            for operation, count in (
                ("get", len(state.stack(line))),
                ("put", len(state.train)),
            ):
                for k in range(1, min(count, self.env.yard.train_count_limit) + 1):
                    try:
                        result.append(
                            self._prepare(
                                state, context, Action(line, operation, k), before_key
                            )
                        )
                    except InvalidAction as exc:
                        if diagnostics is not None:
                            reason = str(exc).split(":", 1)[0]
                            diagnostics[reason] = diagnostics.get(reason, 0) + 1
                        # Prefix lengths and protection only get tighter as k increases.
                        if str(exc).startswith(
                            (
                                "TRAIN_LENGTH",
                                "TRAIN_COUNT",
                                "TRACK_LENGTH",
                                "PROTECTED_VEHICLE",
                                "HOOK_GATE",
                                "ACTION_LIMIT",
                                "NORTH_ROUTE_BLOCKED",
                            )
                        ):
                            break
        return tuple(result)

    def step(self, state, candidate):
        if state != self.state or candidate.before_key != self.decision_key(
            state, self.context
        ):
            raise InvalidAction("STALE_CANDIDATE_OR_SESSION")
        fresh = self.prepare(state, candidate.action, self.context)
        after = fresh.after
        event = {
            "schema_version": 2,
            "hook": after.hook,
            "action": fresh.action.to_dict(),
            "vehicles": fresh.vehicles,
            "route": fresh.route,
            "before_hash": state.fingerprint(),
            "after_hash": after.fingerprint(),
            "before": state.to_dict(),
            "after": after.to_dict(),
            "context_before": self.context.to_dict(),
            "context_after": fresh.after_context.to_dict(),
            "decision_before": fresh.before_key,
            "contract_hash": self.contract_hash,
            "rules": asdict(self.rules),
            "route_model": self.env.route_model,
            "yard_hash": self.env.yard_hash,
            "scenario_hash": self.env.scenario_hash,
            "train_length_mm": self.env.length(after.train)
            + self.env.yard.locomotive_mm,
        }
        self.events.append(event)
        self.state, self.context = after, fresh.after_context
        return after

    def snapshot(self):
        return {
            "state": self.state.to_dict(),
            "business_context": self.context.to_dict(),
            "contract_hash": self.contract_hash,
            "scenario_hash": self.env.scenario_hash,
        }

    def restore(self, snapshot):
        if (
            snapshot.get("contract_hash") != self.contract_hash
            or snapshot.get("scenario_hash") != self.env.scenario_hash
        ):
            raise ValueError("snapshot rules/physical model/request mismatch")
        return self.reset(
            State.from_dict(snapshot["state"]),
            BusinessContext.from_dict(snapshot["business_context"]),
        )
