"""
plasticity.py — Sparse CSR Phase-Timing-Dependent Plasticity (Phase 9)

Key changes over Phase 7:
  - Works directly on CSR edge arrays (no NxN allocation)
  - Uses per-edge delayed phases returned by network.step()
  - atan2 wrapping for correct [-π, π] delta_phi
  - Per-row homeostatic normalization identical to CUDA kernel
"""

import torch
from .config import SYNAPTIC_W_MAX

ETA_PLUS = 0.001  # LTP rate
ETA_MINUS = 0.001  # LTD rate
TAU_PLUS = 0.5  # LTP time constant
TAU_MINUS = 0.5  # LTD time constant


def apply_ptdp(network, delayed_phi_edge: torch.Tensor):
    """
    Apply PTDP in-place to network.weights using CSR layout.

    Args:
      network: LeviathanNetwork (provides row_ptr, col_idx, weights, phi, DA)
      delayed_phi_edge: (E,) tensor — delayed source phase per edge,
                         as returned by network.step() / network._get_delayed_phi_per_edge()
    """
    N = network.num_nodes
    da = float(network.DA)

    for dst in range(N):
        start = int(network.row_ptr[dst])
        end = int(network.row_ptr[dst + 1])
        if end == start:
            continue

        # Δφ = φ_src(t - τ) - φ_dst(t), wrapped to [-π, π]
        phi_del = delayed_phi_edge[start:end]  # (deg,)
        phi_dst = network.phi[dst]
        delta = phi_del - phi_dst
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))

        pos = delta > 0  # j leads i → potentiate
        neg = ~pos  # i leads j → depress

        w = network.weights[start:end]

        # LTP
        if pos.any():
            w[pos] += da * ETA_PLUS * torch.exp(-delta[pos] / TAU_PLUS)

        # LTD (delta is negative here, exp receives positive value)
        if neg.any():
            w[neg] -= ETA_MINUS * torch.exp(delta[neg] / TAU_MINUS)

        # Clamp: weights ≥ 0
        w.clamp_(min=0.0)

        # Homeostatic normalization — row sum must not exceed SYNAPTIC_W_MAX
        row_sum = w.sum()
        if row_sum > SYNAPTIC_W_MAX:
            w *= SYNAPTIC_W_MAX / row_sum

        network.weights[start:end] = w
