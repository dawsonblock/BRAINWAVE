"""
topology.py — Sparse CSR topology builder for Leviathan v2.0

Returns CSR-format edge lists (O(E) memory) instead of dense NxN.
Weights are initialized from Uniform(0, INIT_W_SCALE) for a richer
landscape than binary 0/1 — better reflects weighted connectome intent.
"""

import torch
import math
from .config import REGIONS, TOPOLOGY_LAMBDA, CONDUCTION_VELOCITY, DT, MAX_DELAY_TICKS

INIT_W_SCALE = 0.5  # initial weight max


def generate_spatial_topology(n_nodes: int):
    """
    Distribute nodes across 3D anatomical regions.
    Returns: positions (N,3), base_omegas (N,), region_labels list[str]
    """
    positions = torch.zeros((n_nodes, 3))
    base_omegas = torch.zeros(n_nodes)
    region_labels = []

    nodes_per_region = max(1, n_nodes // len(REGIONS))
    idx = 0
    for name, params in REGIONS.items():
        count = (
            nodes_per_region if name != list(REGIONS.keys())[-1] else (n_nodes - idx)
        )

        r = (params["r_max"] - params["r_min"]) * torch.rand(count) + params["r_min"]
        theta = 2.0 * math.pi * torch.rand(count)

        positions[idx : idx + count, 0] = r * torch.cos(theta)
        positions[idx : idx + count, 1] = r * torch.sin(theta)
        positions[idx : idx + count, 2] = (
            params["z"][1] - params["z"][0]
        ) * torch.rand(count) + params["z"][0]

        base_omegas[idx : idx + count] = torch.clamp(
            torch.normal(
                mean=params["omega"][0], std=params["omega"][1], size=(count,)
            ),
            min=0.1,
        )
        region_labels.extend([name] * count)
        idx += count

    return positions, base_omegas, region_labels


def build_csr_topology(positions: torch.Tensor, region_labels: list):
    """
    Build sparse CSR topology from 3D node positions.

    Connection probability: P(i→j) = exp(-d_ij / lambda)
    Weights: Uniform(0, INIT_W_SCALE)  (non-binary, learnable via PTDP)
    Delays: integer ticks = round(d_ij / (v_c * DT)), clamped to history

    Returns:
      row_ptr   (N+1,)  int32
      col_idx   (E,)    int32
      weights   (E,)    float32
      delay_ticks (E,)  int32
    """
    N = positions.shape[0]

    # Pairwise distances (N, N)
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.linalg.norm(diffs, dim=2)

    # Sample adjacency mask (small-world Bernoulli)
    prob = torch.exp(-distances / TOPOLOGY_LAMBDA)
    prob.fill_diagonal_(0.0)
    adj_mask = torch.bernoulli(prob).bool()

    # Build edge list sorted by row (dst)
    srcs, dsts = adj_mask.nonzero(as_tuple=True)  # src → dst direction

    # We store as dst-indexed CSR (rows = destinations)
    rows = dsts
    cols = srcs
    order = torch.argsort(rows)
    rows, cols = rows[order], cols[order]

    # Weights: Uniform(0, INIT_W_SCALE) — not binary
    E = rows.shape[0]
    weights = torch.rand(E) * INIT_W_SCALE

    # Delays in ticks
    edge_dists = distances[rows, cols]
    delay_secs = edge_dists / CONDUCTION_VELOCITY
    delay_tick_vals = (delay_secs / DT).long().clamp(min=1, max=MAX_DELAY_TICKS - 1)

    # Build row_ptr
    row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    ones = torch.ones(E, dtype=torch.int32)
    row_ptr[1:].scatter_add_(0, rows.int(), ones)
    row_ptr = torch.cumsum(row_ptr, dim=0)

    return (
        row_ptr.int(),
        cols.int(),
        weights.float(),
        delay_tick_vals.int(),
    )
