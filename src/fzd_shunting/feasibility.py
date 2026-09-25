"""Necessary terminal assignment conditions; passing is not a routing certificate."""

from collections import deque


def _flow(edges, source, sink):
    graph = {source: {}, sink: {}}
    for a, b, capacity in edges:
        graph.setdefault(a, {})
        graph.setdefault(b, {})
        graph[a][b] = graph[a].get(b, 0) + capacity
        graph[b].setdefault(a, 0)
    total = 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            a = queue.popleft()
            for b, cap in graph[a].items():
                if cap > 0 and b not in parent:
                    parent[b] = a
                    queue.append(b)
        if sink not in parent:
            return total
        amount = float("inf")
        b = sink
        while parent[b] is not None:
            a = parent[b]
            amount = min(amount, graph[a][b])
            b = a
        b = sink
        while parent[b] is not None:
            a = parent[b]
            graph[a][b] -= amount
            graph[b][a] += amount
            b = a
        total += amount


def terminal_feasibility(dispatcher):
    d, reasons = dispatcher, []
    fixed = {no: line for line, cars in d._locked_suffixes.items() for no in cars}
    for line, cars in d._locked_suffixes.items():
        if d.goals.assignment(cars, line) is None:
            reasons.append("IMMOVABLE_SUFFIX_TARGET: " + line)
    count_edges, length_edges, slots, eligible = [], [], set(), {}
    source, sink = ("source",), ("sink",)
    for no, car in d.env.cars.items():
        v = ("vehicle", no)
        count_edges.append((source, v, 1))
        length_edges.append((source, v, car.length_mm))
        choices = 0
        for target in car.targets:
            line = target.line
            if no in fixed and fixed[no] != line:
                continue
            if car.length_mm > d.env.tracks[line].final_capacity_mm:
                continue
            bound = d.goals.bounds[line]
            domain = (
                (car.initial_position,)
                if no in d.protected
                else target.positions or range(1, bound + 1)
            )
            positions = [p for p in domain if 1 <= p <= bound]
            if not positions:
                continue
            choices += 1
            eligible.setdefault(line, []).append(car.length_mm)
            length_edges.append((v, ("line", line), car.length_mm))
            for p in positions:
                slot = ("slot", line, p)
                slots.add(slot)
                count_edges.append((v, slot, 1))
        if not choices:
            reasons.append("NO_FEASIBLE_TARGET: " + no)
    for slot in slots:
        count_edges.append((slot, ("line", slot[1]), 1))
    for line, lengths in eligible.items():
        cap = d.env.tracks[line].final_capacity_mm
        count_edges.append(
            (("line", line), sink, min(d.goals.bounds[line], cap // min(lengths)))
        )
        length_edges.append((("line", line), sink, cap))
    if not reasons:
        if _flow(count_edges, source, sink) < len(d.env.cars):
            reasons.append("TARGET_SLOT_OR_COUNT_ASSIGNMENT")
        if _flow(length_edges, source, sink) < sum(
            c.length_mm for c in d.env.cars.values()
        ):
            reasons.append("TARGET_LENGTH_ASSIGNMENT_RELAXATION")
    return dict(
        status="infeasible" if reasons else "not_disproved",
        reasons=reasons,
        scope="Necessary slot/count and fractional-length assignment; no complete-route proof.",
    )
