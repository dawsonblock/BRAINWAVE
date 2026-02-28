import torch
import math
from .config import REGIONS, TOPOLOGY_LAMBDA


def generate_spatial_topology(n_nodes: int):
    """
    Distributes nodes randomly across the defined 3D regions.
    Returns:
    - positions: (N, 3) tensor
    - base_omegas: (N,) tensor of natural frequencies
    - region_labels: list of strings of length N
    """
    positions = torch.zeros((n_nodes, 3))
    base_omegas = torch.zeros(n_nodes)
    region_labels = []

    # Assign roughly equal nodes to each region for now
    nodes_per_region = max(1, n_nodes // len(REGIONS))

    idx = 0
    for name, params in REGIONS.items():
        count = (
            nodes_per_region if name != list(REGIONS.keys())[-1] else (n_nodes - idx)
        )

        # Random r, theta for xy plane
        r = (params["r_max"] - params["r_min"]) * torch.rand(count) + params["r_min"]
        theta = 2.0 * math.pi * torch.rand(count)

        positions[idx : idx + count, 0] = r * torch.cos(theta)  # x
        positions[idx : idx + count, 1] = r * torch.sin(theta)  # y
        positions[idx : idx + count, 2] = (
            params["z"][1] - params["z"][0]
        ) * torch.rand(count) + params["z"][0]

        # Frequencies
        base_omegas[idx : idx + count] = torch.normal(
            mean=params["omega"][0], std=params["omega"][1], size=(count,)
        )
        base_omegas[idx : idx + count] = torch.clamp(
            base_omegas[idx : idx + count], min=0.1
        )

        region_labels.extend([name] * count)
        idx += count

    return positions, base_omegas, region_labels


def calculate_conduction_delays(
    positions: torch.Tensor, conduction_velocity: float
) -> torch.Tensor:
    """
    Computes Euclidean distance between all nodes to form delay matrix tau.
    Returns:
    - delay_matrix (seconds): (N, N)
    """
    # Broadcasting to get pairwise differences
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    distances = torch.linalg.norm(diffs, dim=2)  # (N, N)
    delay_matrix = distances / conduction_velocity
    return delay_matrix


def initialize_adjacency_matrix(
    positions: torch.Tensor, lambda_decay: float
) -> torch.Tensor:
    """
    Creates the Small-World adjacency matrix based on P = kappa * exp(-d/lambda)
    """
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.linalg.norm(diffs, dim=2)

    # Let kappa = 1 for relative probability
    prob_matrix = torch.exp(-distances / lambda_decay)

    # Prevent self-connections
    prob_matrix.fill_diagonal_(0)

    # Sample Bernoulli
    adjacency = torch.bernoulli(prob_matrix)
    return adjacency
