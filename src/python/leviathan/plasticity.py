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

    # Neuromodulatory Metamodulation: DA scales LTP (ETA_PLUS)
    # High DA = high plasticity for rewarding trajectories.
    # Low DA = stable weights.
    effective_eta_plus = ETA_PLUS * da
    effective_eta_minus = ETA_MINUS  # LTD stays relatively constant

    # Compute weight updates for ALL edges at once
    dw = torch.zeros_like(network.weights)
    if pos.any():
        dw[pos] = effective_eta_plus * torch.exp(-delta[pos] / TAU_PLUS)
    if neg.any():
        dw[neg] = -effective_eta_minus * torch.exp(delta[neg] / TAU_MINUS)

    # 4. Apply Update (In-place)
    with torch.no_grad():
        network.weights.add_(dw)
        network.weights.clamp_(-1.5, 1.5)

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


def apply_active_inf_learning(network, delayed_phi_edge: torch.Tensor, lr=0.0001):
    """
    Active Inference Learning: Update weights to minimize prediction error.
    dw ∝ -error_dst * sin(phi_src_del - phi_dst)
    """
    error_dst = network.error[network.dst_idx]
    phi_dst = network.phi[network.dst_idx]

    # Gradient of squared error w.r.t coupling:
    # E = 0.5 * error^2
    # dE/dw = error * d(phi - phi_target)/dw
    # In our simplified dynamics, dphi/dw ∝ sin(phi_src_del - phi_dst)
    error_dst = network.error[network.dst_idx]
    phi_dst = network.phi[network.dst_idx]

    # Gradient of squared error w.r.t coupling:
    # E = 0.5 * error^2
    # dE/dw = error * d(phi - phi_target)/dw
    # In our simplified dynamics, dphi/dw ∝ sin(phi_src_del - phi_dst)
    grad = error_dst * torch.sin(delayed_phi_edge - phi_dst)

    # ACH scales the learning rate for prediction errors (attention-weighted learning)
    ach = float(network.ACH)
    effective_lr = lr * ach

    dw = -effective_lr * grad

    # Dale's Principle Sign Enforcement
    pos_mask = (network.weights > 0).float()
    neg_mask = (network.weights < 0).float()

    network.weights.add_(dw)

    # Clamp based on original sign
    network.weights.mul_(pos_mask).clamp_(min=0.0)
    network.weights.add_(network.weights.mul_(neg_mask).clamp_(max=0.0))


def apply_synaptic_scaling(network, target_atp=500.0, rate=0.001):
    """
    Homeostatic Synaptic Scaling: Scale all incoming weights to maintain target ATP.
    If ATP is low, scale weights down. If ATP is high, scale weights up.
    """
    # Inverse metabolic stress factor
    # stress = 1.0 means ATP is at target. stress > 1.0 means ATP is low.
    stress = target_atp / network.atp.clamp(min=1.0)

    # Scaling factor per node (destination)
    # We want to multiply weights by (1.0 - delta)
    # delta is positive if stress is high (ATP low)
    delta = rate * (stress - 1.0)
    scaling = (1.0 - delta).clamp(min=0.5, max=1.5)

    # Sparse gather: apply scaling[dst] to each weight
    network.weights.mul_(scaling[network.dst_idx])
