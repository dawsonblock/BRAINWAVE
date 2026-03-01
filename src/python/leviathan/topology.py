"""
topology.py — Phase 10: Spatial-bin graph builder (O(N) average, not O(N²))

Old code built a full (N,N) distance matrix. This replaces it with spatial
binning: only nodes in neighbouring bins are candidates, reducing the
connectivity-sampling cost from O(N²) to O(N · k) where k = mean candidates
per bin (~constant for uniform distributions).

The public API `build_csr_topology(positions, region_labels)` is preserved so
run_agent.py / LeviathanNetwork don't need changes.
"""

import torch
import math
import random
from .config import (
    TOPOLOGY_LAMBDA,
    CONDUCTION_VELOCITY,
    DT,
    MAX_DELAY_TICKS,
    INHIBITORY_FRACTION,
    E_I_STRENGTH_RATIO,
)

INIT_W_SCALE = 0.5


# ─── kept for backward compat (used by render scripts etc.) ──────────────────
def generate_spatial_topology(n_nodes: int):
    """
    Distribute nodes across anatomical regions.
    Returns: positions (N,3), base_omegas (N,), region_labels list[str]
    """
    from .config import REGIONS

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


# ─── Primary builder (CSR, spatial-bin, O(N·k)) ──────────────────────────────
def build_csr_topology(positions: torch.Tensor, region_labels: list):
    """
    Spatial-bin CSR builder.
    Avoids full (N,N) distance matrix by only checking nodes in adjacent bins.

    Connection probability: P(i→j) = exp(-d_ij / lambda_conn)
    Weights: Uniform(0, INIT_W_SCALE)
    Delays: integer ticks clamped to [1, MAX_DELAY_TICKS-1]

    Returns: row_ptr (N+1,), col_idx (E,), weights (E,), delay_ticks (E,)
    """
    N = positions.shape[0]
    lam = TOPOLOGY_LAMBDA
    v_c = CONDUCTION_VELOCITY
    dt = DT

    # Use XY projection for binning (ignore Z for bin assignment)
    xy = positions[:, :2]
    mins = xy.min(dim=0).values
    xy_n = xy - mins  # shift so all ≥ 0

    bin_size = lam  # bin ≈ one decay length
    bx_arr = (xy_n[:, 0] / bin_size).long()
    by_arr = (xy_n[:, 1] / bin_size).long()

    # --- Phase 11: Spatial Tiling (Re-ordering for Locality) ---
    # Sort nodes by bin index to ensure spatial neighbors are memory neighbors
    bin_idx = bx_arr * 1000 + by_arr  # simple 2D hash for sorting
    order = torch.argsort(bin_idx)

    positions = positions[order]
    region_labels = [region_labels[i] for i in order.tolist()]

    # Re-calculate bin markers for the new order
    xy = positions[:, :2]
    xy_n = xy - xy.min(dim=0).values
    bx_arr = (xy_n[:, 0] / bin_size).long()
    by_arr = (xy_n[:, 1] / bin_size).long()

    # E/I Assignment
    is_inhibitory = torch.rand(N) < INHIBITORY_FRACTION
    bins: dict[tuple, list] = {}
    for i in range(N):
        key = (int(bx_arr[i]), int(by_arr[i]))
        bins.setdefault(key, []).append(i)

    # Collect edges (dst, src) with spatial-bin sampling
    dst_list, src_list, w_list, d_list = [], [], [], []

    for dst in range(N):
        bx, by = int(bx_arr[dst]), int(by_arr[dst])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates = bins.get((bx + dx, by + dy), [])
                for src in candidates:
                    if src == dst:
                        continue
                    dist = float(torch.norm(positions[dst] - positions[src]))
                    if random.random() < math.exp(-dist / lam):
                        dst_list.append(dst)
                        src_list.append(src)

                        # Dale's Principle: Excitatory vs Inhibitory
                        w = random.random() * INIT_W_SCALE
                        if is_inhibitory[src]:
                            w = -w * E_I_STRENGTH_RATIO
                        w_list.append(w)

                        ticks = max(1, min(int(dist / (v_c * dt)), MAX_DELAY_TICKS - 1))
                        d_list.append(ticks)

    if not dst_list:
        # Fallback: ring connections so the network is never isolated
        for i in range(N):
            dst_list.append(i)
            src_list.append((i + 1) % N)
            w_list.append(INIT_W_SCALE / 2)
            d_list.append(1)

    E = len(dst_list)
    dsts = torch.tensor(dst_list, dtype=torch.long)
    srcs = torch.tensor(src_list, dtype=torch.long)
    ws = torch.tensor(w_list, dtype=torch.float32)
    ds = torch.tensor(d_list, dtype=torch.int32)

    # Sort by destination row
    order = torch.argsort(dsts)
    dsts, srcs, ws, ds = dsts[order], srcs[order], ws[order], ds[order]

    # Build row_ptr via scatter
    row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    row_ptr[1:].scatter_add_(0, dsts.int(), torch.ones(E, dtype=torch.int32))
    row_ptr = torch.cumsum(row_ptr, dim=0)

    return (
        row_ptr.int(),
        srcs.int(),
        ws.float(),
        ds.int(),
        is_inhibitory,
        positions,
        region_labels,
    )


def save_topology_bin(path: str, topo_data: tuple):
    """
    Save Phase 12 topology to binary for CUDA import.
    Format:
      [int32] N, E
      [int32 x N+1] row_ptr
      [int32 x E]   col_idx
      [float32 x E] weights
      [int32 x E]   delay_ticks
      [float32 x N] base_omegas
      [bool x N]    is_inhibitory
      [bool x N]    is_cortex (from region_labels)
    """
    import struct

    (
        row_ptr,
        col_idx,
        weights,
        delay_ticks,
        is_inhibitory,
        positions,
        region_labels,
    ) = topo_data
    N = len(row_ptr) - 1
    E = len(col_idx)

    # Simplified omegas for export parity
    omegas = torch.ones(N) * 1.0

    is_cortex = torch.tensor(
        [label.startswith("cortex") for label in region_labels], dtype=torch.bool
    )

    with open(path, "wb") as f:
        f.write(struct.pack("ii", N, E))
        f.write(row_ptr.numpy().astype("int32").tobytes())
        f.write(col_idx.numpy().astype("int32").tobytes())
        f.write(weights.numpy().astype("float32").tobytes())
        f.write(delay_ticks.numpy().astype("int32").tobytes())
        f.write(omegas.numpy().astype("float32").tobytes())
        f.write(is_inhibitory.numpy().astype("bool").tobytes())
        f.write(is_cortex.numpy().astype("bool").tobytes())
    print(f"Saved Phase 12 topology to {path} (N={N}, E={E})")


def build_distance_graph(
    N: int,
    space_size: float = 10.0,
    lambda_conn: float = 2.0,
    max_delay_ticks: int = 16,
):
    """
    Standalone spatial-bin graph builder (no region structure).
    Used by scale_runner.py and arena_generalization.py.
    Returns: row_ptr, col_idx, weights, delay_ticks
    """
    pos = torch.rand(N, 2) * space_size
    fake_3d = torch.cat([pos, torch.zeros(N, 1)], dim=1)
    labels = ["cortex"] * N
    # Temporarily monkey-patch TOPOLOGY_LAMBDA via local scope
    import leviathan.config as cfg

    _old = cfg.TOPOLOGY_LAMBDA
    cfg.TOPOLOGY_LAMBDA = lambda_conn
    out = build_csr_topology(fake_3d, labels)
    cfg.TOPOLOGY_LAMBDA = _old
    return out
