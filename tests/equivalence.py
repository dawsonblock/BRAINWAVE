"""
tests/equivalence.py — Phase 9E: Python-only determinism + CSR correctness test.

Without a GPU available, this verifies:
  1. Same seed → identical trajectories (determinism)
  2. CSR coupling produces finite, non-NaN values (correctness)
  3. V_MAX clamping holds (stability)
  4. PTDP updates don't explode weights (plasticity health)

Run:
  PYTHONPATH=src/python python3 tests/equivalence.py
"""

import sys
import math
import random
import numpy as np
import torch

sys.path.insert(0, "src/python")

from leviathan.network import LeviathanNetwork
from leviathan.plasticity import apply_ptdp
from leviathan.endocrine import update_metabolism, update_neuromodulators

N = 64
TICKS = 200
SEED = 42
V_MAX = 50.0  # must match network.V_MAX


def _seed_all(seed: int):
    """Seed everything that topology.py / network.py uses."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_sim(seed: int, ptdp: bool = True):
    _seed_all(seed)
    net = LeviathanNetwork(N)
    sensory = torch.zeros(N)

    for _ in range(TICKS):
        delayed = net.step(external_sensory_input=sensory)
        if ptdp:
            apply_ptdp(net, delayed)
        update_metabolism(net)
        update_neuromodulators(net)

    return net.phi.clone(), net.phi_dot.clone(), net.weights.clone()


def test_determinism():
    """Same seed must yield identical final phases."""
    phi_a, _, _ = run_sim(SEED)
    phi_b, _, _ = run_sim(SEED)
    assert torch.allclose(
        phi_a, phi_b, atol=0.0
    ), f"Determinism FAILED — max diff = {(phi_a - phi_b).abs().max():.2e}"
    print("[PASS] Determinism: identical trajectories for same seed")


def test_finiteness():
    """All values must be finite after 200 ticks."""
    phi, phi_dot, weights = run_sim(SEED)
    assert torch.isfinite(phi).all(), "[FAIL] phi contains NaN/Inf"
    assert torch.isfinite(phi_dot).all(), "[FAIL] phi_dot contains NaN/Inf"
    assert torch.isfinite(weights).all(), "[FAIL] weights contains NaN/Inf"
    print("[PASS] Finiteness: no NaN/Inf in phi, phi_dot, weights")


def test_velocity_clamp():
    """phi_dot magnitude must never exceed V_MAX."""
    _, phi_dot, _ = run_sim(SEED)
    vmax_obs = phi_dot.abs().max().item()
    assert vmax_obs <= V_MAX + 1e-5, f"[FAIL] V_MAX exceeded: {vmax_obs:.4f} > {V_MAX}"
    print(f"[PASS] Velocity clamp: max |phi_dot| = {vmax_obs:.4f} ≤ {V_MAX}")


def test_weight_bounds():
    """Weights must stay in [0, SYNAPTIC_W_MAX] from config."""
    from leviathan.config import SYNAPTIC_W_MAX

    _, _, weights = run_sim(SEED)
    assert (weights >= 0).all(), "[FAIL] Negative weights found"
    assert (
        weights <= SYNAPTIC_W_MAX + 1e-4
    ).all(), f"[FAIL] Weight exceeds SYNAPTIC_W_MAX: {weights.max():.4f}"
    print(f"[PASS] Weights: all in [0, {SYNAPTIC_W_MAX}]")


def test_phase_range():
    """Phases must stay in [0, 2π)."""
    phi, _, _ = run_sim(SEED)
    lo, hi = phi.min().item(), phi.max().item()
    assert lo >= -1e-5, f"[FAIL] phi below 0: {lo:.6f}"
    assert hi < 2 * math.pi + 1e-5, f"[FAIL] phi above 2π: {hi:.6f}"
    print(f"[PASS] Phase range: [{lo:.4f}, {hi:.4f}] ⊆ [0, 2π)")


def test_ptdp_vs_frozen():
    """PTDP-enabled sim should reach a different weight distribution than frozen."""
    _, _, w_on = run_sim(SEED, ptdp=True)
    _, _, w_off = run_sim(SEED, ptdp=False)
    diff = (w_on - w_off).abs().max().item()
    assert diff > 1e-6, f"[FAIL] PTDP had no effect on weights (diff={diff:.2e})"
    print(f"[PASS] PTDP effect: max weight diff from frozen = {diff:.4f}")


if __name__ == "__main__":
    print(f"\nLeviathan v2.0 — Phase 9E Equivalence Suite  (N={N}, ticks={TICKS})\n")
    test_determinism()
    test_finiteness()
    test_velocity_clamp()
    test_weight_bounds()
    test_phase_range()
    test_ptdp_vs_frozen()
    print("\n✅  All tests passed.")
