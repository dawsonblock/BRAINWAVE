"""
arena_generalization.py — Phase 10: Multi-seed generalization benchmark

Measures mean ± std of food eaten across train/test splits of arena seeds.
Replaces single-run "266 food" with a statistically meaningful metric.

Usage:
  PYTHONPATH=src/python python3 src/python/experiments/arena_generalization.py
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "src/python")

from leviathan.network import LeviathanNetwork
from leviathan.plasticity import apply_ptdp
from leviathan.endocrine import update_metabolism, update_neuromodulators


TICKS_PER_SEED = 1000
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [5, 6, 7, 8, 9]
N_NODES = 100


def run_seed(seed: int, ptdp: bool = True) -> int:
    """Run one arena episode and return food eaten."""
    torch.manual_seed(seed)

    try:
        from environment.arena import Arena

        arena = Arena(seed=seed)
        use_arena = True
    except Exception:
        use_arena = False

    net = LeviathanNetwork(N_NODES)
    sensory = torch.zeros(N_NODES)
    eaten = 0

    for t in range(TICKS_PER_SEED):
        delayed = net.step(external_sensory_input=sensory)
        if ptdp:
            apply_ptdp(net, delayed)
        update_metabolism(net)
        update_neuromodulators(net)

        if use_arena:
            try:
                sensory_raw, food_eaten, done = arena.step(net.phi.numpy())
                sensory = torch.tensor(sensory_raw, dtype=torch.float32)
                if food_eaten:
                    eaten += 1
            except Exception:
                pass

    return eaten


def evaluate():
    print(f"\nGeneralization Benchmark  (N={N_NODES}, ticks={TICKS_PER_SEED})\n")

    train_scores = [run_seed(s) for s in TRAIN_SEEDS]
    test_scores = [run_seed(s) for s in TEST_SEEDS]

    train_arr = np.array(train_scores, dtype=float)
    test_arr = np.array(test_scores, dtype=float)

    print(f"Train seeds {TRAIN_SEEDS}: {train_scores}")
    print(f"  mean = {train_arr.mean():.1f} ± {train_arr.std():.1f}")
    print(f"\nTest  seeds {TEST_SEEDS}: {test_scores}")
    print(f"  mean = {test_arr.mean():.1f} ± {test_arr.std():.1f}")

    gap = train_arr.mean() - test_arr.mean()
    print(f"\nGeneralization gap: {gap:.2f} (lower is better)")

    if abs(gap) < train_arr.std():
        print("✅  Train ≈ Test — no significant overfitting to map geometry.")
    else:
        print("⚠️   Gap exceeds train std — possible map-specific bias.")

    return {
        "train_mean": float(train_arr.mean()),
        "train_std": float(train_arr.std()),
        "test_mean": float(test_arr.mean()),
        "test_std": float(test_arr.std()),
        "gap": float(gap),
    }


if __name__ == "__main__":
    results = evaluate()
    print("\nSummary:", results)
