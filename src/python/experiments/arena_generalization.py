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
import random

sys.path.insert(0, "src/python")

from leviathan.network import LeviathanNetwork
from leviathan.motor import MotorDecoder
from leviathan.plasticity import apply_ptdp
from leviathan.endocrine import update_metabolism, update_neuromodulators
from leviathan.config import MOTOR_GAIN, DA_REWARD_SPIKE, ATP_FOOD_REWARD, SENSORY_GAIN


TICKS_PER_SEED = 1000
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [5, 6, 7, 8, 9]
N_NODES = 200


def run_seed(seed: int, ptdp: bool = True) -> int:
    """Run one arena episode and return food eaten."""
    torch.manual_seed(seed)
    # Arena internally seeds random.seed(seed)
    from environment.arena import Arena

    arena = Arena(seed=seed)

    net = LeviathanNetwork(N_NODES)
    motor_decoder = MotorDecoder(network=net)
    sensory = torch.zeros(N_NODES)
    eaten = 0

    for t in range(TICKS_PER_SEED):
        # ---- SENSORY PHASE ----
        food_prox, _ = arena.get_sensory_raycasts(
            num_rays=5, fov_deg=120, max_dist=250.0
        )
        for i, prox in enumerate(food_prox):
            if i < N_NODES:
                sensory[i] = SENSORY_GAIN * float(prox)

        # ---- COGNITIVE PHASE ----
        delayed = net.step(external_sensory_input=sensory)
        if ptdp:
            apply_ptdp(net, delayed)
        update_metabolism(net)
        update_neuromodulators(net)

        # ---- MOTOR PHASE ----
        raw_l, raw_r = motor_decoder.decode_torque()
        torque_l = raw_l * MOTOR_GAIN + 2.0
        torque_r = raw_r * MOTOR_GAIN + 2.0
        food_eaten = arena.move_agent(torque_l, torque_r)

        if food_eaten > 0:
            eaten += food_eaten
            net.atp += ATP_FOOD_REWARD * food_eaten
            net.DA = min(net.DA + DA_REWARD_SPIKE * food_eaten, 10.0)

    return eaten


def evaluate():
    print(f"\nGeneralization Benchmark  (N={N_NODES}, ticks={TICKS_PER_SEED})\n")

    train_scores = []
    for s in TRAIN_SEEDS:
        score = run_seed(s)
        print(f"  Seed {s} : {score} food")
        train_scores.append(score)

    test_scores = []
    for s in TEST_SEEDS:
        score = run_seed(s)
        print(f"  Seed {s} : {score} food")
        test_scores.append(score)

    train_arr = np.array(train_scores, dtype=float)
    test_arr = np.array(test_scores, dtype=float)

    print(f"\nResults:")
    print(f"  Train: {train_arr.mean():.1f} ± {train_arr.std():.1f}")
    print(f"  Test : {test_arr.mean():.1f} ± {test_arr.std():.1f}")

    gap = train_arr.mean() - test_arr.mean()
    print(f"\nGeneralization gap: {gap:.2f} (lower is better)")

    if abs(gap) < (train_arr.std() + 0.1):
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
