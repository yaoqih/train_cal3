"""Pure physical transitions. Route precision is explicitly limited to topology/occupancy."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json

from .domain import Action, InvalidAction, Scenario, State, Yard


class NorthEndRouter:
    model = "north-end-occupancy-v1"
    capabilities = {
        "verified": [
            "north_access",
            "intermediate_track_occupancy",
            "prefix_suffix",
            "length_count",
        ],
        "not_modelled": [
            "switch_combinations",
            "reversal_clearance",
            "continuous_train_occupancy",
        ],
    }

    def __init__(self, yard: Yard):
        self.north = {t.name: set() for t in yard.tracks}
        self.south = {t.name: set() for t in yard.tracks}
        for south, north in yard.edges:
            self.north[south].add(north)
            self.south[north].add(south)
        self.graph = {n: self.north[n] | self.south[n] for n in self.north}
        self._route = lru_cache(maxsize=4096)(self._route)

    def route(self, state: State, destination: str):
        occupied = frozenset(name for name, cars in state.stacks if cars)
        return self._route(state.loco_line, state.loco_end, occupied, destination)

    def _route(self, loco_line, loco_end, occupied, destination):
        if loco_line == destination and loco_end == "North":
            return (destination,)
        starts = set((self.north if loco_end == "North" else self.south)[loco_line])
        finishes = self.north[destination]
        if loco_line not in occupied:
            starts.add(loco_line)
        queue = deque(
            (node, (loco_line,) if node == loco_line else (loco_line, node))
            for node in sorted(starts)
            if node not in occupied
        )
        visited = set(node for node, _ in queue)
        while queue:
            node, path = queue.popleft()
            if node in finishes:
                return path + (destination,)
            for nxt in sorted(self.graph[node]):
                if nxt not in visited and nxt not in occupied:
                    visited.add(nxt)
                    queue.append((nxt, path + (nxt,)))
        raise InvalidAction("NORTH_ROUTE_BLOCKED: " + destination)


class Environment:
    def __init__(self, yard: Yard, scenario: Scenario, router=None):
        self.yard = yard
        self.scenario = scenario
        self.tracks = yard.by_name
        self.cars = scenario.cars
        if router is None and yard.route_model != NorthEndRouter.model:
            raise ValueError("no registered router for model: " + yard.route_model)
        self.router = router or NorthEndRouter(yard)
        if not getattr(self.router, "model", None):
            raise ValueError("router must declare its model version")
        if not isinstance(getattr(self.router, "capabilities", None), dict):
            raise ValueError(
                "router must declare its verified capabilities and boundaries"
            )
        self.route_model = self.router.model
        self.capabilities = self.router.capabilities
        self.length = lru_cache(maxsize=16384)(self.length)
        self.yard_hash = hashlib.sha256(
            json.dumps(asdict(yard), sort_keys=True).encode()
        ).hexdigest()
        scenario_data = asdict(scenario)
        scenario_data.pop("name")
        self.scenario_hash = hashlib.sha256(
            json.dumps(scenario_data, sort_keys=True).encode()
        ).hexdigest()

    def length(self, ids):
        return sum(self.cars[no].length_mm for no in ids)

    def validate_state(self, state: State):
        errors = []
        seen = list(state.train)
        if len(state.stacks) != len(self.tracks) or set(dict(state.stacks)) != set(
            self.tracks
        ):
            errors.append("STATE_TRACKS: track set differs from yard")
        if type(state.hook) is not int or state.hook < 0:
            errors.append("HOOK_COUNT")
        for line, ids in state.stacks:
            seen.extend(ids)
            if line not in self.tracks:
                errors.append("UNKNOWN_TRACK: " + line)
            elif ids and not self.tracks[line].parking:
                errors.append("TRANSIT_OCCUPIED: " + line)
            elif (
                set(ids) <= self.cars.keys()
                and self.length(ids) > self.tracks[line].length_mm
            ):
                errors.append(
                    "TRACK_LENGTH: %s (%d > %d mm)"
                    % (line, self.length(ids), self.tracks[line].length_mm)
                )
        if len(seen) != len(set(seen)) or set(seen) != set(self.cars):
            errors.append("VEHICLE_CONSERVATION: duplicate, missing or unknown vehicle")
        if len(state.train) > self.yard.train_count_limit:
            errors.append("TRAIN_COUNT")
        if set(state.train) <= self.cars.keys() and (
            self.length(state.train) + self.yard.locomotive_mm
            > self.yard.train_limit_mm
        ):
            errors.append("TRAIN_LENGTH")
        if state.loco_line not in self.tracks or state.loco_end not in (
            "North",
            "South",
        ):
            errors.append("LOCO_LOCATION")
        return errors

    def transition(self, state: State, action: Action, *, validated=False):
        if not validated:
            errors = self.validate_state(state)
            if errors:
                raise InvalidAction("INVALID_STATE: " + "; ".join(errors))
        if action.line not in self.tracks or not self.tracks[action.line].parking:
            raise InvalidAction("ACTION_TRACK")
        if action.operation not in ("get", "put"):
            raise InvalidAction("ACTION_OPERATION")
        if type(action.count) is not int or action.count <= 0:
            raise InvalidAction("ACTION_COUNT")
        stack = state.stack(action.line)
        if action.operation == "get":
            if action.count > len(stack):
                raise InvalidAction("GET_PREFIX_COUNT")
            block = stack[: action.count]
            new_stack = stack[action.count :]
            train = state.train + block
        else:
            if action.count > len(state.train):
                raise InvalidAction("PUT_SUFFIX_COUNT")
            block = state.train[-action.count :]
            new_stack = block + stack
            train = state.train[: -action.count]
        # Check the full arriving train AND the full departing train.
        if max(len(state.train), len(train)) > self.yard.train_count_limit:
            raise InvalidAction("TRAIN_COUNT")
        if (
            max(self.length(state.train), self.length(train)) + self.yard.locomotive_mm
            > self.yard.train_limit_mm
        ):
            raise InvalidAction("TRAIN_LENGTH")
        if self.length(new_stack) > self.tracks[action.line].length_mm:
            raise InvalidAction("TRACK_LENGTH: " + action.line)
        route = self.router.route(state, action.line)
        stacks = tuple(
            (line, new_stack if line == action.line else cars)
            for line, cars in state.stacks
        )
        after = replace(
            state,
            stacks=stacks,
            train=train,
            loco_line=action.line,
            loco_end="North",
            hook=state.hook + 1,
        )
        return after, block, route
