"""
scale_runner.py — Phase 10: 100k+ node execution path

Tests that the sparse engine can handle large N without OOM or NaN.
Uses build_distance_graph for fast topology generation.

Usage:
  PYTHONPATH=src/python python3 src/python/experiments/scale_runner.py --n 10000
  PYTHONPATH=src/python python3 src/python/experiments/scale_runner.py --n 100000
"""

import sys
import time
import argparse
import torch

sys.path.insert(0, "src/python")

from leviathan.topology import build_distance_graph
from leviathan.network import LeviathanNetwork
from leviathan.scale_config import ScaleConfig, MemoryBudget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000, help="Number of nodes")
    parser.add_argument("--ticks", type=int, default=100, help="Ticks to simulate")
    args = parser.parse_args()

    N = args.n
    TICKS = args.ticks

    print(f"\nScale Runner — N={N}, ticks={TICKS}")
    print("─" * 50)

    # Build topology
    t0 = time.time()
    row_ptr, col_idx, weights, delays = build_distance_graph(N)
    E = col_idx.shape[0]
    print(f"Topology built in {time.time()-t0:.2f}s  (E={E})")

    # Memory estimate
    budget = MemoryBudget.estimate(N, E)
    print(
        f"Estimated memory: {budget['total_MB']:.0f}MB  "
        f"(history={budget['history_MB']:.0f}MB, edges={budget['edges_MB']:.0f}MB)"
    )
    print(
        f"Fits 8GB GPU: {'✓' if budget['fits_8GB_GPU'] else '✗'}  "
        f"Fits 24GB GPU: {'✓' if budget['fits_24GB_GPU'] else '✗'}"
    )

    # Build network (uses internal topology for now — TODO: accept CSR)
    print("\nBuilding LeviathanNetwork...")
    t0 = time.time()
    net = LeviathanNetwork(N)
    print(
        f"Network init in {time.time()-t0:.2f}s  " f"(E_actual={net.col_idx.shape[0]})"
    )

    sensory = torch.zeros(N)

    # Run
    print(f"\nRunning {TICKS} ticks...")
    t0 = time.time()
    for tick in range(TICKS):
        net.step(external_sensory_input=sensory)

    elapsed = time.time() - t0
    rate = TICKS / elapsed

    print(f"Done in {elapsed:.2f}s  ({rate:.1f} ticks/s)")
    print(f"Final phi mean: {net.phi.mean():.4f}")
    print(f"Final phi NaN: {torch.isnan(net.phi).any()}")


if __name__ == "__main__":
    main()
