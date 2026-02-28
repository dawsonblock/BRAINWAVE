"""
gpu_learning.py — Phase 10: Vectorized GPU-side PTDP

Full PyTorch vectorization of the PTDP update over all edges simultaneously.
Works on CPU or GPU (moves with the tensors).
No Python loop — single tensor pass over (E,) edge arrays.
"""

import torch
from .config import SYNAPTIC_W_MAX

ETA_PLUS = 0.001
ETA_MINUS = 0.001
TAU_PLUS = 0.5
TAU_MINUS = 0.5


def gpu_ptdp_step(
    phi: torch.Tensor,  # (N,)
    history_phi: torch.Tensor,  # (MAX_DELAY, N)
    dst_idx: torch.Tensor,  # (E,) long
    col_idx: torch.Tensor,  # (E,) long  — source nodes
    weights: torch.Tensor,  # (E,) float (in-place update)
    delay_ticks: torch.Tensor,  # (E,) long
    hist_idx: int,
    MAX_DELAY: int,
    da: float = 1.0,  # Dopamine gate
) -> None:
    """
    In-place vectorized PTDP over all edges.
    Δw = DA * η+ * exp(-Δφ / τ+)  if Δφ > 0 (j leads i)
       = -η- * exp(Δφ / τ-)       if Δφ ≤ 0 (i leads j)
    """
    slots = (hist_idx - delay_ticks) % MAX_DELAY
    phi_del = history_phi[slots, col_idx]  # (E,) delayed source phase
    phi_dst = phi[dst_idx]  # (E,) current destination phase

    delta = torch.atan2(
        torch.sin(phi_del - phi_dst), torch.cos(phi_del - phi_dst)
    )  # wrapped to [-π, π]

    pos = delta > 0
    neg = ~pos

    dw = torch.zeros_like(weights)
    dw[pos] = da * ETA_PLUS * torch.exp(-delta[pos] / TAU_PLUS)
    dw[neg] = -ETA_MINUS * torch.exp(delta[neg] / TAU_MINUS)

    weights.add_(dw)
    weights.clamp_(min=0.0)

    # Homeostatic per-row normalization via scatter
    N = phi.shape[0]
    row_sum = torch.zeros(N, device=weights.device)
    row_sum.scatter_add_(0, dst_idx, weights)

    scale = (row_sum > SYNAPTIC_W_MAX).float() * (
        SYNAPTIC_W_MAX / row_sum.clamp(min=1e-6)
    )
    scale = torch.where(row_sum > SYNAPTIC_W_MAX, scale, torch.ones_like(scale))
    weights.mul_(scale[dst_idx])
