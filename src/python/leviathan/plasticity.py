"""
plasticity.py — Fully Vectorized Sparse CSR Phase-Timing-Dependent Plasticity (Phase 10)

Key changes:
  - Eliminated O(N) loop over nodes. Entire update is now O(E) via tensor masks.
  - Vectorized homeostatic normalization using scatter_add.
  - Identical logic to CUDA ptdp.cu kernel.
"""

import torch
from .config import SYNAPTIC_W_MAX

ETA_PLUS = 0.001
ETA_MINUS = 0.001
TAU_PLUS = 0.5
TAU_MINUS = 0.5


def apply_ptdp(network, delayed_phi_edge: torch.Tensor):
    """
    Apply PTDP in-place to network.weights using fully vectorized CSR layout.

    Args:
      network: LeviathanNetwork (provides row_ptr, dst_idx, weights, phi, DA)
      delayed_phi_edge: (E,) tensor — delayed source phase per edge.
    """
    da = float(network.DA)
    phi_dst = network.phi[network.dst_idx]  # (E,)

    # Δφ = φ_src(t - τ) - φ_dst(t), wrapped to [-π, π]
    delta = torch.atan2(
        torch.sin(delayed_phi_edge - phi_dst), torch.cos(delayed_phi_edge - phi_dst)
    )

    pos = delta > 0
    neg = ~pos

    # Compute weight updates for ALL edges at once
    dw = torch.zeros_like(network.weights)
    if pos.any():
        dw[pos] = da * ETA_PLUS * torch.exp(-delta[pos] / TAU_PLUS)
    if neg.any():
        dw[neg] = -ETA_MINUS * torch.exp(delta[neg] / TAU_MINUS)

    network.weights.add_(dw)
    network.weights.clamp_(min=0.0)

    # Vectorized homeostatic per-row normalization
    # row_sum[i] = Σ weight_ij for all j in incoming(i)
    row_sum = torch.zeros(network.num_nodes, device=network.weights.device)
    row_sum.scatter_add_(0, network.dst_idx, network.weights)

    # Scale only the rows that exceed MAX
    scale = torch.ones_like(row_sum)
    over_mask = row_sum > SYNAPTIC_W_MAX
    scale[over_mask] = SYNAPTIC_W_MAX / row_sum[over_mask].clamp(min=1e-6)

    # Apply per-row scale to each edge's weight
    network.weights.mul_(scale[network.dst_idx])
