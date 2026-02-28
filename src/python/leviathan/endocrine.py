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
    Drains ATP based on the cost equation. Prunes highly expensive synapses when near death.
    """
    # ATP Cost Equation
    # dE/dt = -(R_basal + k_cost * |phi_dot| + k_maint * sum(W_incoming))

    velocity_cost = K_COST * torch.abs(network.phi_dot)
    synaptic_cost = K_MAINT * network.weights.sum(dim=1)

    drain = (R_BASAL + velocity_cost + synaptic_cost) * DT

    network.atp -= drain

    # Check for metabolic death/pruning
    # If a node's ATP drops below 0, we prune its weakest incoming synapse
    # to reduce its `k_maint` cost.
    starving_nodes = torch.where(network.atp <= ATP_DEATH_THRESHOLD)[0]

    for i in starving_nodes:
        # Give it a small emergency ATP boost to survive the tick
        network.atp[i] = 10.0

        # Find weakest non-zero synapse
        incoming_w = network.weights[i, :]
        active_idx = torch.where(incoming_w > 0)[0]

        if len(active_idx) > 0:
            weakest_idx = active_idx[torch.argmin(incoming_w[active_idx])]
            # Prune it (set to 0 permanently)
            network.weights[i, weakest_idx] = 0.0


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
