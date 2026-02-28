import torch
from .config import A_PLUS, A_MINUS, TAU_PLUS, TAU_MINUS, SYNAPTIC_W_MAX, DT

def apply_ptdp(weights: torch.Tensor, current_phi: torch.Tensor, delayed_phi: torch.Tensor, dopamine: float):
    """
    Applies Phase-Timing-Dependent Plasticity to the weight matrix in place.
    
    delta_w = DA * A+ * e^(-delta_phi / tau+) if delta_phi > 0 (j leads i)
    delta_w = -A- * e^(delta_phi / tau-)     if delta_phi <= 0 (i leads j)
    """
    N = weights.shape[0]
    
    # delayed_phi[dst, src] is phi_src(t - tau)
    # current_phi[dst] is phi_dst(t)
    
    current_phi_matrix = current_phi.unsqueeze(1).expand(N, N)
    
    # Delta Phase: Does the signal from j arrive BEFORE i fires?
    # To simplify continuous phase, we look at the raw difference mod 2pi
    # But for PTDP causality, we want to know relative lead/lag.
    # An approximation for continuous:
    # If sin(delayed_phi - current_phi) > 0, delayed_phi is "ahead" in the cycle.
    
    # Let's use the explicit delta from the spec:
    delta_phi = delayed_phi - current_phi_matrix
    
    # Map to [-pi, pi]
    delta_phi = (delta_phi + torch.pi) % (2.0 * torch.pi) - torch.pi
    
    # Positive delta (j leads i)
    mask_pos = (delta_phi > 0).float()
    dw_pos = mask_pos * dopamine * A_PLUS * torch.exp(-delta_phi / TAU_PLUS)
    
    # Negative delta (i leads j)
    mask_neg = (delta_phi <= 0).float()
    dw_neg = mask_neg * -A_MINUS * torch.exp(delta_phi / TAU_MINUS) # delta_phi is negative, so exp(neg)
    
    delta_w = (dw_pos + dw_neg) * DT
    
    # Update only existing weights (topology is fixed, only strengths change)
    active_synapses = (weights > 0).float()
    weights += delta_w * active_synapses
    
    # Clamp weights > 0
    weights.clamp_(min=0.0)
    
    # Homeostatic Normalization
    # Sum of incoming weights to node i cannot exceed SYNAPTIC_W_MAX
    incoming_sum = weights.sum(dim=1, keepdim=True)
    
    # Avoid div by zero
    incoming_sum[incoming_sum == 0] = 1.0
    
    # Find nodes exceeding max
    scale_factor = SYNAPTIC_W_MAX / incoming_sum
    scale_factor = torch.clamp(scale_factor, max=1.0) # Only scale down, don't scale up
    
    weights *= scale_factor
    
    return weights
