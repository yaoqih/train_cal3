"""Relational graph policy with container context and grouped joint actions."""

from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NODE_DIM = 29
ACTION_DIM = 12
RELATIONS = 16
FEATURE_SCHEMA = "container-boundary-after-order-budget-v3"
CHECKPOINT_SCHEMA = 4


@dataclass
class DecisionGraph:
    nodes: np.ndarray
    edges: np.ndarray
    relations: np.ndarray
    action_features: np.ndarray
    loco: int
    sequences: tuple
    containers: np.ndarray
    vehicle_containers: np.ndarray
    action_nodes: np.ndarray
    affected: np.ndarray
    owners: np.ndarray
    group_sizes: np.ndarray
    post_vehicles: np.ndarray
    post_positions: np.ndarray
    post_owners: np.ndarray
    post_counts: np.ndarray
    action_posts: np.ndarray


@dataclass
class GraphBatch:
    nodes: torch.Tensor
    edges: torch.Tensor
    relations: torch.Tensor
    action_features: torch.Tensor
    loco: torch.Tensor
    node_graph: torch.Tensor
    node_counts: torch.Tensor
    action_graph: torch.Tensor
    action_nodes: torch.Tensor
    affected: torch.Tensor
    owners: torch.Tensor
    affected_counts: torch.Tensor
    sequence_indices: torch.Tensor
    sequence_output: torch.Tensor
    ordered_nodes: torch.Tensor
    container_indices: torch.Tensor
    container_mask: torch.Tensor
    vehicle_containers: torch.Tensor
    group_sizes: torch.Tensor
    post_vehicles: torch.Tensor
    post_positions: torch.Tensor
    post_owners: torch.Tensor
    post_counts: torch.Tensor
    action_posts: torch.Tensor
    action_mask: torch.Tensor
    action_slots: torch.Tensor
    sequence_lengths: list

    def to(self, device):
        return GraphBatch(
            **{
                f.name: (
                    getattr(self, f.name).to(device, non_blocking=True)
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )

    def pin_memory(self):
        return GraphBatch(
            **{
                f.name: (
                    getattr(self, f.name).pin_memory()
                    if isinstance(getattr(self, f.name), torch.Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


def collate_graphs(graphs):
    if not graphs:
        raise ValueError("cannot batch an empty graph list")
    ns = np.asarray([len(g.nodes) for g in graphs], dtype=np.int64)
    acts = np.asarray([len(g.action_features) for g in graphs], dtype=np.int64)
    offsets = np.cumsum(np.r_[0, ns[:-1]])
    ao = np.cumsum(np.r_[0, acts[:-1]])
    ps = np.asarray([len(g.post_counts) for g in graphs], dtype=np.int64)
    po = np.cumsum(np.r_[0, ps[:-1]])
    sequences = [seq + off for g, off in zip(graphs, offsets) for seq in g.sequences]
    lengths = [len(seq) for seq in sequences]
    width = max(lengths, default=0)
    seq_indices = np.zeros((len(sequences), width), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq_indices[i, : len(seq)] = seq
    ordered = np.concatenate(sequences) if sequences else np.empty(0, dtype=np.int64)
    seq_output = (
        np.concatenate([i * width + np.arange(n) for i, n in enumerate(lengths)])
        if lengths
        else np.empty(0, dtype=np.int64)
    )
    mask = np.arange(max(1, int(acts.max())))[None, :] < acts[:, None]
    cw = max(len(g.containers) for g in graphs)
    ci = np.zeros((len(graphs), cw), dtype=np.int64)
    cm = np.zeros((len(graphs), cw), dtype=bool)
    for i, (g, off) in enumerate(zip(graphs, offsets)):
        ci[i, : len(g.containers)] = g.containers + off
        cm[i, : len(g.containers)] = True
    owners = np.concatenate([g.owners + off for g, off in zip(graphs, ao)])
    arrays = dict(
        nodes=np.concatenate([g.nodes for g in graphs]),
        edges=np.concatenate(
            [g.edges + off for g, off in zip(graphs, offsets)], axis=1
        ),
        relations=np.concatenate([g.relations for g in graphs]),
        action_features=np.concatenate([g.action_features for g in graphs]),
        loco=np.asarray([g.loco + off for g, off in zip(graphs, offsets)]),
        node_graph=np.repeat(np.arange(len(graphs)), ns),
        node_counts=ns,
        action_graph=np.repeat(np.arange(len(graphs)), acts),
        action_nodes=np.concatenate(
            [
                np.where(g.action_nodes < 0, -1, g.action_nodes + off)
                for g, off in zip(graphs, offsets)
            ]
        ),
        affected=np.concatenate([g.affected + off for g, off in zip(graphs, offsets)]),
        owners=owners,
        affected_counts=np.bincount(owners, minlength=int(acts.sum())),
        sequence_indices=seq_indices,
        sequence_output=seq_output,
        ordered_nodes=ordered,
        container_indices=ci,
        container_mask=cm,
        vehicle_containers=np.concatenate(
            [g.vehicle_containers + off for g, off in zip(graphs, offsets)]
        ),
        group_sizes=np.concatenate([g.group_sizes for g in graphs]),
        post_vehicles=np.concatenate(
            [g.post_vehicles + off for g, off in zip(graphs, offsets)]
        ),
        post_positions=np.concatenate([g.post_positions for g in graphs]),
        post_owners=np.concatenate([g.post_owners + off for g, off in zip(graphs, po)]),
        post_counts=np.concatenate([g.post_counts for g in graphs]),
        action_posts=np.concatenate(
            [
                np.where(g.action_posts < 0, -1, g.action_posts + off)
                for g, off in zip(graphs, po)
            ]
        ),
        action_mask=mask,
        action_slots=np.flatnonzero(mask),
    )
    return GraphBatch(
        **{k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()},
        sequence_lengths=lengths,
    )


class GraphEncoder:
    """Request-static facts are compiled once; each observation fills dynamic facts."""

    def __init__(self, dispatcher):
        env = dispatcher.env
        self.names, self.ids = tuple(env.tracks), tuple(env.cars)
        self.tracks = {n: i for i, n in enumerate(self.names)}
        self.vehicles = {n: len(self.names) + i for i, n in enumerate(self.ids)}
        self.loco = len(self.names) + len(self.ids)
        slots = sorted(
            {
                (t.line, p)
                for car in env.cars.values()
                for t in car.targets
                for p in t.positions
            }
        )
        self.slots = {s: self.loco + 1 + i for i, s in enumerate(slots)}
        f = np.zeros((self.loco + 1 + len(slots), NODE_DIM), dtype=np.float32)
        edges, relations = [], []

        def edge(a, b, r):
            edges.append((a, b))
            relations.append(r)

        limit = env.yard.train_limit_mm - env.yard.locomotive_mm
        for name, ti in self.tracks.items():
            t = env.tracks[name]
            f[ti, 0] = 1
            f[ti, 11] = float(t.inner)
            f[ti, 12] = float(t.kind == "temporary")
            f[ti, 17:21] = (
                t.length_mm / limit,
                t.final_capacity_mm / limit,
                dispatcher.goals.bounds.get(name, 0) / env.yard.train_count_limit,
                float(name in dispatcher.protected_lines),
            )
        for south, north in env.yard.edges:
            edge(self.tracks[south], self.tracks[north], 0)
            edge(self.tracks[north], self.tracks[south], 1)
        for no, car in env.cars.items():
            vi = self.vehicles[no]
            f[vi, 1] = 1
            f[vi, 4] = car.length_mm / limit
            f[vi, 7:10] = (car.heavy, car.closed_door, no in dispatcher.protected)
            f[vi, 17] = float(car.weigh)
            f[vi, 20] = car.initial_position / 64 if no in dispatcher.protected else 0
            for target in car.targets:
                ti = self.tracks[target.line]
                edge(vi, ti, 4)
                edge(ti, vi, 5)
                for p in target.positions:
                    si = self.slots[(target.line, p)]
                    edge(vi, si, 8)
                    edge(si, vi, 9)
            if no in dispatcher.protected:
                si = self.slots[(car.initial_line, car.initial_position)]
                edge(vi, si, 14)
                edge(si, vi, 15)
        for (line, pos), si in self.slots.items():
            f[si, 3], f[si, 10] = 1, pos / 64
            edge(si, self.tracks[line], 10)
            edge(self.tracks[line], si, 11)
        f[self.loco, 2] = 1
        self.features, self.edges, self.relations = f, tuple(edges), tuple(relations)
        self.containers = np.asarray([*self.tracks.values(), self.loco], dtype=np.int64)

    def encode(self, d, state, candidates, context, hook_budget):
        env = d.env
        limit = env.yard.train_limit_mm - env.yard.locomotive_mm
        count_limit = env.yard.train_count_limit
        f, edges, relations = (
            self.features.copy(),
            list(self.edges),
            list(self.relations),
        )
        f[:, 15] = state.hook / 100
        # A fixed scale distinguishes a 40-hook and a 64-hook request even at hook 0.
        f[:, 28] = max(0, hook_budget - state.hook) / 64

        def edge(a, b, r):
            edges.append((a, b))
            relations.append(r)

        sequences, owners = [], []
        for line, stack in (*state.stacks, ("__loco__", state.train)):
            ti = self.loco if line == "__loco__" else self.tracks[line]
            if line != "__loco__":
                f[ti, 4] = env.length(stack) / max(1, env.tracks[line].length_mm)
                f[ti, 5] = len(stack) / count_limit
                for op, col, qc in (("get", 13, 24), ("put", 16, 25)):
                    f[ti, col] = (
                        max(0, d.rules.earliest(line, op) - state.hook - 1) / 100
                    )
                    quota = d.remaining_quota(line, op, context)
                    f[ti, qc] = -1 if quota is None else quota / 20
                delivered, suffix = d.goals.line_metrics(stack, line)
                f[ti, 26:28] = (delivered / count_limit, suffix / count_limit)
            seq = np.asarray([self.vehicles[n] for n in stack], dtype=np.int64)
            if stack:
                sequences.append(seq)
                owners.extend([ti] * len(stack))
            for i, no in enumerate(stack):
                vi = self.vehicles[no]
                f[vi, 6] = (i + 1) / 50
                f[vi, 14] = float(env.cars[no].target(line) is not None)
                f[vi, 21:24] = (i == 0, i == len(stack) - 1, (len(stack) - i - 1) / 50)
                edge(ti, vi, 2)
                edge(vi, ti, 3)
            for a, b in zip(seq, seq[1:]):
                edge(a, b, 6)
                edge(b, a, 7)
        f[self.loco, 5] = len(state.train) / count_limit
        f[self.loco, 16] = (limit - env.length(state.train)) / limit
        f[self.loco, 18] = env.length(state.train) / limit
        f[self.loco, 19] = float(state.loco_end == "North")
        edge(self.loco, self.tracks[state.loco_line], 12)
        edge(self.tracks[state.loco_line], self.loco, 13)
        af, an, affected, ao, keys, ap = [], [], [], [], [], []
        post_ids, post_vehicles, post_positions, post_owners, post_counts = (
            {},
            [],
            [],
            [],
            [],
        )

        def post(stack):
            if not stack:
                return -1
            if stack not in post_ids:
                index = len(post_ids)
                post_ids[stack] = index
                post_counts.append(len(stack))
                for i, no in enumerate(stack):
                    post_vehicles.append(self.vehicles[no])
                    post_positions.append(
                        (
                            (i + 1) / len(stack),
                            (len(stack) - i - 1) / len(stack),
                            len(stack) / count_limit,
                        )
                    )
                    post_owners.append(index)
            return post_ids[stack]

        def vehicle(no):
            return self.vehicles[no] if no is not None else -1

        for i, c in enumerate(candidates):
            get = c.action.operation == "get"
            line = c.action.line
            before = state.stack(line)
            after = c.after.stack(line)
            source = self.tracks[line] if get else self.loco
            dest = self.loco if get else self.tracks[line]
            source_after = after if get else c.after.train
            dest_after = c.after.train if get else after
            exposed = (
                (source_after[0] if get else source_after[-1]) if source_after else None
            )
            contact = (
                (state.train[-1] if state.train else None)
                if get
                else (before[0] if before else None)
            )
            an.append(
                [
                    source,
                    dest,
                    vehicle(c.vehicles[0]),
                    vehicle(c.vehicles[-1]),
                    vehicle(exposed),
                    vehicle(contact),
                ]
            )
            af.append(
                [
                    float(get),
                    float(not get),
                    c.action.count / count_limit,
                    c.gain / max(1, len(self.ids)),
                    c.destination_count / c.action.count,
                    len(c.route) / max(1, len(self.names)),
                    env.length(c.after.train) / limit,
                    float(line in d.protected_lines),
                    c.delivered_gain / max(1, len(self.ids)),
                    c.suffix_gain / max(1, len(self.ids)),
                    env.length(c.vehicles) / limit,
                    len(after) / count_limit,
                ]
            )
            affected.extend(self.vehicles[n] for n in c.vehicles)
            ao.extend([i] * len(c.vehicles))
            keys.append((line, get))
            ap.append([post(source_after), post(dest_after)])
        from collections import Counter

        counts = Counter(keys)
        return DecisionGraph(
            f,
            np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
            np.asarray(relations, dtype=np.int64),
            np.asarray(af, dtype=np.float32).reshape(-1, ACTION_DIM),
            self.loco,
            tuple(sequences),
            self.containers,
            np.asarray(owners, dtype=np.int64),
            np.asarray(an, dtype=np.int64).reshape(-1, 6),
            np.asarray(affected, dtype=np.int64),
            np.asarray(ao, dtype=np.int64),
            np.asarray([counts[k] for k in keys], dtype=np.float32),
            np.asarray(post_vehicles, dtype=np.int64),
            np.asarray(post_positions, dtype=np.float32).reshape(-1, 3),
            np.asarray(post_owners, dtype=np.int64),
            np.asarray(post_counts, dtype=np.int64),
            np.asarray(ap, dtype=np.int64).reshape(-1, 2),
        )


def encode_graph(dispatcher, state, candidates, context=None, hook_budget=64):
    if hook_budget < 1:
        raise ValueError("positive global hook budget required")
    context = dispatcher.resolve_context(state, context)
    if not hasattr(dispatcher, "_graph_encoder"):
        dispatcher._graph_encoder = GraphEncoder(dispatcher)
    return dispatcher._graph_encoder.encode(
        dispatcher, state, candidates, context, hook_budget
    )


class GraphPolicy(nn.Module):
    def __init__(self, hidden=96, layers=3, hook_budget=64):
        super().__init__()
        if hidden < 2 or hidden % 2 or layers < 1 or hook_budget < 1:
            raise ValueError(
                "positive layers/budget and positive even hidden size required"
            )
        self.hidden, self.layers, self.hook_budget = hidden, layers, hook_budget
        self.profile = None
        self.encoder = nn.Linear(NODE_DIM, hidden)
        self.relations = nn.Embedding(RELATIONS, hidden)
        self.messages = nn.ModuleList(
            nn.Linear(hidden * 2, hidden) for _ in range(layers)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.LayerNorm(hidden)
            )
            for _ in range(layers)
        )
        self.order_encoder = nn.GRU(
            hidden, hidden // 2, batch_first=True, bidirectional=True
        )
        self.order_update = nn.Linear(hidden * 2, hidden)
        self.container_update = nn.Linear(hidden * 2, hidden)
        self.container_attention = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.container_norm = nn.LayerNorm(hidden)
        self.post_encoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden * 10 + ACTION_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def bind(self, dispatcher):
        if self.profile is not None and self.profile != dispatcher.profile:
            raise ValueError("policy physical/business contract mismatch")
        self.profile = dict(dispatcher.profile)

    def validate_profile(self, dispatcher):
        if self.profile is None or self.profile != dispatcher.profile:
            raise ValueError(
                "checkpoint does not match the physical/business rules; train matching weights"
            )

    def forward_batch(self, batch):
        h = torch.relu(self.encoder(batch.nodes))
        source, destination = batch.edges
        degree = h.new_zeros((len(h), 1), dtype=torch.float32)
        degree.index_add_(0, destination, degree.new_ones((len(destination), 1)))
        relation = self.relations(batch.relations)
        for message, update in zip(self.messages, self.updates):
            msg = torch.relu(message(torch.cat((h[source], relation), dim=1)))
            aggregated = h.new_zeros((len(h), self.hidden), dtype=torch.float32)
            aggregated.index_add_(0, destination, msg.float())
            h = update(torch.cat((h, aggregated / degree.clamp_min(1)), dim=1))
        if batch.sequence_lengths:
            packed = pack_padded_sequence(
                h[batch.sequence_indices],
                batch.sequence_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            encoded, _ = self.order_encoder(packed)
            ordered, _ = pad_packed_sequence(encoded, batch_first=True)
            ordered = ordered.flatten(0, 1)[batch.sequence_output]
            updated = torch.relu(
                self.order_update(torch.cat((h[batch.ordered_nodes], ordered), dim=1))
            )
            h = h.index_copy(0, batch.ordered_nodes, updated.to(h.dtype))
        inventory = h.new_zeros((len(h), self.hidden), dtype=torch.float32)
        inventory.index_add_(
            0, batch.vehicle_containers, h[batch.ordered_nodes].float()
        )
        counts = torch.bincount(batch.vehicle_containers, minlength=len(h)).clamp_min(1)
        ci, cm = batch.container_indices, batch.container_mask
        containers = torch.relu(
            self.container_update(
                torch.cat((h[ci], (inventory / counts[:, None])[ci]), dim=-1)
            )
        )
        attended, _ = self.container_attention(
            containers, containers, containers, key_padding_mask=~cm, need_weights=False
        )
        containers = self.container_norm(containers + attended)
        h = h.index_copy(0, ci[cm], containers[cm].to(h.dtype))
        global_state = h.new_zeros((len(batch.loco), self.hidden), dtype=torch.float32)
        global_state.index_add_(0, batch.node_graph, h.float())
        global_state = global_state / batch.node_counts[:, None]
        affected = h.new_zeros(
            (len(batch.action_nodes), self.hidden), dtype=torch.float32
        )
        affected.index_add_(0, batch.owners, h[batch.affected].float())
        affected = affected / batch.affected_counts[:, None].clamp_min(1)
        post = h.new_zeros(
            (len(batch.post_counts) + 1, self.hidden), dtype=torch.float32
        )
        encoded_post = self.post_encoder(
            torch.cat((h[batch.post_vehicles], batch.post_positions), dim=1)
        )
        post.index_add_(0, batch.post_owners + 1, encoded_post.float())
        post = (
            post
            / torch.cat((batch.post_counts.new_ones(1), batch.post_counts))[:, None]
        )

        def pick(indices):
            return h[indices.clamp_min(0)] * (indices >= 0)[:, None]

        src, dst, head, tail, exposed, contact = batch.action_nodes.T
        # log-mean-exp group score + conditional count softmax equals raw - log(group size).
        representations = torch.cat(
            (
                global_state[batch.action_graph],
                pick(src),
                pick(dst),
                affected,
                pick(head),
                pick(tail),
                pick(exposed),
                pick(contact),
                post[batch.action_posts[:, 0] + 1],
                post[batch.action_posts[:, 1] + 1],
                batch.action_features,
            ),
            dim=1,
        )
        logits = self.actor(representations).flatten().float() - batch.group_sizes.log()
        padded = logits.new_full((batch.action_mask.numel(),), float("-inf"))
        return padded.scatter(0, batch.action_slots, logits).reshape(
            batch.action_mask.shape
        )

    def forward(self, dispatcher, state, candidates, context=None, hook_budget=None):
        graph = encode_graph(
            dispatcher,
            state,
            candidates,
            context,
            self.hook_budget if hook_budget is None else hook_budget,
        )
        return self.forward_batch(
            collate_graphs([graph]).to(next(self.parameters()).device)
        )[0, : len(candidates)]

    def log_probabilities(
        self, dispatcher, state, candidates, context=None, hook_budget=None
    ):
        self.validate_profile(dispatcher)
        with torch.inference_mode():
            return (
                torch.log_softmax(
                    self(dispatcher, state, candidates, context, hook_budget), dim=0
                )
                .cpu()
                .tolist()
            )

    def choose(self, dispatcher, state, candidates, context=None):
        scores = self.log_probabilities(dispatcher, state, candidates, context)
        return candidates[max(range(len(candidates)), key=lambda i: scores[i])]


def load_model(path, device="cpu"):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (
        cp.get("schema_version") != CHECKPOINT_SCHEMA
        or cp.get("feature_schema") != FEATURE_SCHEMA
    ):
        raise ValueError(
            "expected grouped-relative policy checkpoint schema 4; train new weights"
        )
    model = GraphPolicy(cp["hidden"], cp["layers"], cp["hook_budget"])
    model.load_state_dict(cp["state_dict"])
    model.profile = cp["profile"]
    return model.to(device).eval()
