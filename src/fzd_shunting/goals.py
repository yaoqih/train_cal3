"""Target semantics shared by execution, augmentation and learning."""

from functools import lru_cache

GOAL_SEMANTICS = "relative-slots-protected-anchor-v2"


def slot_bounds(scenario, yard):
    result = {t.name: t.reference_slots for t in yard.tracks if t.parking}
    counts = {name: 0 for name in result}
    for car in scenario.vehicles:
        for target in car.targets:
            counts[target.line] += 1
            result[target.line] = max(
                result[target.line], max(target.positions, default=0)
            )
    for name in result:
        result[name] = max(result[name], counts[name])
    result.update(dict(scenario.inner_capacities))
    return result


class GoalEvaluator:
    def __init__(self, env):
        self.env = env
        self.bounds = slot_bounds(env.scenario, env.yard)
        self.protected = env.scenario.protected
        self.assignment = lru_cache(maxsize=8192)(self.assignment)
        self.line_metrics = lru_cache(maxsize=8192)(self.line_metrics)

    def assignment(self, ids, line):
        if self.env.length(ids) > self.env.tracks[line].final_capacity_mm:
            return None
        bound, previous, result = self.bounds.get(line, 0), 0, []
        for no in ids:
            car = self.env.cars[no]
            target = car.target(line)
            if target is None:
                return None
            domain = (
                (car.initial_position,) if no in self.protected else target.positions
            )
            selected = (
                next((p for p in sorted(domain) if previous < p <= bound), None)
                if domain
                else previous + 1
            )
            if selected is None or selected > bound:
                return None
            previous = selected
            result.append((no, selected))
        return tuple(result)

    def line_metrics(self, ids, line):
        delivered = sum(self.env.cars[no].target(line) is not None for no in ids)
        suffix = next(
            (
                len(ids) - i
                for i in range(len(ids) + 1)
                if self.assignment(ids[i:], line) is not None
            ),
            0,
        )
        return delivered, suffix

    def metrics(self, state):
        delivered = settled = complete = 0
        for line, ids in state.stacks:
            a, b = self.line_metrics(ids, line)
            delivered += a
            settled += b
            if self.assignment(ids, line) is not None:
                complete += len(ids)
        return {
            "delivered": delivered,
            "ordered_suffix": settled,
            "complete_lines_vehicles": complete,
        }

    def errors(self, state):
        errors = ["LOCO_NOT_EMPTY"] if state.train else []
        errors.extend(
            "TARGET_OR_ORDER: " + line
            for line, ids in state.stacks
            if self.assignment(ids, line) is None
        )
        return errors
