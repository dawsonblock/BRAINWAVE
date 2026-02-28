import torch
from .network import LeviathanNetwork
from .config import (
    INITIAL_ATP,
    R_BASAL,
    K_COST,
    K_MAINT,
    ATP_DEATH_THRESHOLD,
    DT,
    TAU_DA,
    TAU_NA,
    TAU_SER,
    TAU_ACH,
    ALPHA_DA,
    ALPHA_NA,
    ALPHA_SER,
    ALPHA_ACH,
    NA_BASAL,
    ATP_TARGET,
    THETA_NOVEL,
)


def update_metabolism(network: LeviathanNetwork):
    """
    Drains ATP based on the cost equation. Prunes expensive synapses when starving.
    ATP cost:  dE/dt = -(R_basal + k_cost * |phi_dot| + k_maint * sum_of_all_weights/N)
    """
    velocity_cost = K_COST * torch.abs(network.phi_dot)

    # CSR: network.weights is 1D (E,).  Approximate per-node maintenance cost
    # as global_mean_weight (avoids per-row sum which requires row_ptr iteration).
    # For exact per-node cost, use prune_weakest_starving() below.
    mean_weight = network.weights.mean().item() if network.weights.numel() > 0 else 0.0
    synaptic_cost = torch.full((network.num_nodes,), K_MAINT * mean_weight)

    drain = (R_BASAL + velocity_cost + synaptic_cost) * DT
    network.atp -= drain

    # Starving nodes: emergency ATP + prune weakest incoming synapse
    starving_nodes = torch.where(network.atp <= ATP_DEATH_THRESHOLD)[0]
    for i in starving_nodes:
        network.atp[i] = 10.0

    # Delegate CSR-aware pruning to the network
    network.prune_weakest_starving()


def update_neuromodulators(network: LeviathanNetwork):
    """
    Dynamically updates global DA, NA, SER, and ACH levels based on network physiological states.
    """
    mean_atp = network.atp.mean().item()

    # Simple artificial proxy for "damage" or "error"
    # We will use the number of starving nodes as an error signal
    starving_count = (network.atp <= (ATP_DEATH_THRESHOLD + 10.0)).sum().item()
    error_signal = starving_count / float(network.num_nodes)

    # Average magnitude of phase velocity in Cortex (Simulated here as global for proof-of-concept)
    mean_velocity = torch.abs(network.phi_dot).mean().item()

    # DA: Driven by increases in system energy relative to target (Simplified Goal alignment)
    da_drive = ALPHA_DA * max(0.0, mean_atp - ATP_TARGET)
    dDA = (-network.DA / TAU_DA + da_drive) * DT
    network.DA = max(0.0, network.DA + dDA)

    # NA: Driven by Error/Threat (Starvation)
    na_drive = ALPHA_NA * error_signal
    dNA = (-(network.NA - NA_BASAL) / TAU_NA + na_drive) * DT
    network.NA = max(0.0, network.NA + dNA)

    # SER: Stabilized by healthy ATP
    ser_drive = ALPHA_SER * (mean_atp - ATP_TARGET) / ATP_TARGET
    dSER = (-(network.SER - 1.0) / TAU_SER + ser_drive) * DT
    network.SER = max(0.1, network.SER + dSER)  # Clamp minimum scaling to 0.1

    # ACH: Driven by novelty (unexpectedly fast velocities)
    ach_drive = ALPHA_ACH * max(0.0, mean_velocity - THETA_NOVEL)
    dACH = (-(network.ACH - 1.0) / TAU_ACH + ach_drive) * DT
    network.ACH = max(0.1, network.ACH + dACH)  # Cannot drive inertia effectively to 0

    # Clamp all neuromodulators to biologically plausible range
    network.DA = min(max(network.DA, 0.0), 10.0)
    network.NA = min(max(network.NA, 0.0), 5.0)
    network.SER = min(max(network.SER, 0.1), 5.0)
    network.ACH = min(max(network.ACH, 0.1), 5.0)
