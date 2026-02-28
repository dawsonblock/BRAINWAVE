"""
foraging_task.py — Phase 10: Structured embodied reward task scaffold

Used by arena_generalization.py and scale_runner.py.
Separates task definition from network implementation.
"""

import torch
import numpy as np


class ForagingTask:
    """
    Structured foraging: agent must move toward nearest food cluster.
    Reward = 1/(1 + dist_to_nearest_food).
    Foods respawn after being consumed.
    """

    def __init__(
        self, arena_size: float = 10.0, n_food: int = 20, seed: int | None = None
    ):
        self.arena_size = arena_size
        self.n_food = n_food
        if seed is not None:
            torch.manual_seed(seed)
        self.food = torch.rand(n_food, 2) * arena_size
        self.eaten = 0

    def reward(self, agent_pos: torch.Tensor) -> float:
        """Continuous proximity reward."""
        dists = torch.norm(self.food - agent_pos.unsqueeze(0), dim=1)
        return float(1.0 / (1.0 + dists.min()))

    def try_eat(self, agent_pos: torch.Tensor, eat_radius: float = 0.5) -> bool:
        """Returns True and respawns food if agent is within eat_radius."""
        dists = torch.norm(self.food - agent_pos.unsqueeze(0), dim=1)
        idx = dists.argmin()
        if dists[idx] < eat_radius:
            self.food[idx] = torch.rand(2) * self.arena_size
            self.eaten += 1
            return True
        return False

    def respawn_all(self):
        self.food = torch.rand(self.n_food, 2) * self.arena_size


class MultiMapEvaluator:
    """
    Evaluate a network across multiple arena seeds.
    Returns mean ± std of food eaten across seeds.
    """

    def __init__(self, n_maps: int = 10, steps_per_map: int = 2000):
        self.n_maps = n_maps
        self.steps_per_map = steps_per_map

    def evaluate(self, run_fn) -> dict:
        """
        run_fn(seed) → food_count (int)
        Returns {"mean": float, "std": float, "per_seed": list}
        """
        scores = [run_fn(seed) for seed in range(self.n_maps)]
        arr = np.array(scores, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_seed": scores,
        }
