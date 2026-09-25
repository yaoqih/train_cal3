"""Whole-plan relative objective with explicit failed-termination semantics."""


def trajectory_score(dispatcher, state, context, hook_budget, stop_reason=None):
    if hook_budget < 1 or not 0 <= state.hook <= hook_budget:
        raise ValueError("trajectory hook count must be within the total-plan budget")
    if dispatcher.done(state, context):
        return 2.0 - state.hook / (hook_budget + 1)
    metrics = dispatcher.goals.metrics(state)
    n = max(1, len(dispatcher.env.cars))
    progress = (0.55 * metrics["delivered"] + 0.45 * metrics["ordered_suffix"]) / n
    penalties = {None: 0.0, "hook_budget": 0.0, "cycle": 0.25, "dead_end": 0.35}
    if stop_reason not in penalties:
        raise ValueError("unknown failed termination reason")
    # A premature failure does not earn a bonus for using fewer hooks.
    return -1.0 + progress - penalties[stop_reason]
