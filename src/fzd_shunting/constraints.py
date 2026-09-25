"""Composable pure business checks; stateful facts live in BusinessContext."""

from .domain import InvalidAction


class Constraint:
    rule_id = "constraint"
    version = 1

    def before(self, dispatcher, state, context, action):
        pass

    def after(self, dispatcher, state, context, action, after, block):
        pass

    def advance(self, dispatcher, context, action):
        return context

    def terminal(self, dispatcher, state, context):
        return ()


class AccessConstraint(Constraint):
    rule_id = "hook_access"

    def before(self, dispatcher, state, context, action):
        if state.hook + 1 < dispatcher.rules.earliest(action.line, action.operation):
            raise InvalidAction("HOOK_GATE: " + action.line)


class ProtectionConstraint(Constraint):
    rule_id = "initial_protection"

    def after(self, dispatcher, state, context, action, after, block):
        if action.operation == "get" and dispatcher.protected.intersection(block):
            raise InvalidAction("PROTECTED_VEHICLE")
        if action.operation == "put" and action.line in dispatcher.protected_lines:
            if (
                dispatcher.order_assignment(after.stack(action.line), action.line)
                is None
            ):
                raise InvalidAction("PROTECTED_INNER_NO_BUFFER")


class ActionLimitConstraint(Constraint):
    rule_id = "action_limits"

    def before(self, dispatcher, state, context, action):
        for limit in dispatcher.rules.action_limits:
            if action.line in limit.lines and action.operation in limit.operations:
                if context.count(limit.name) >= limit.max_actions:
                    raise InvalidAction("ACTION_LIMIT: " + limit.name)

    def advance(self, dispatcher, context, action):
        return context.increment(
            limit.name
            for limit in dispatcher.rules.action_limits
            if action.line in limit.lines and action.operation in limit.operations
        )


DEFAULT_CONSTRAINTS = (
    AccessConstraint(),
    ProtectionConstraint(),
    ActionLimitConstraint(),
)
