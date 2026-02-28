"""
export_topology.py — Topology exporter for CUDA

Generates CSR topology using LeviathanNetwork, then exports to binary file
readable by the upgraded main.cu (Phase 9D).

Binary format (little-endian):
  [int32]  N
  [int32]  E
  [int32 × (N+1)]  row_ptr
  [int32 × E]      col_idx
  [float32 × E]    weights
  [int32 × E]      delay_ticks
  [float32 × N]    base_omegas

Usage:
  PYTHONPATH=src/python python3 src/python/export_topology.py --n 200 --out leviathan_topology.bin
"""

import sys
import struct
import argparse
import torch

sys.path.insert(0, "src/python")
from leviathan.network import LeviathanNetwork


def export(n_nodes: int, out_path: str):
    print(f"[export] Building topology for N={n_nodes}...")
    net = LeviathanNetwork(n_nodes)

    N = n_nodes
    E = net.col_idx.shape[0]

    print(f"[export] N={N}  E={E}  writing to {out_path}")

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<ii", N, E))

        # row_ptr (N+1, int32)
        row_ptr_np = net.row_ptr.numpy().astype("int32")
        f.write(row_ptr_np.tobytes())

        # col_idx (E, int32)
        col_idx_np = net.col_idx.numpy().astype("int32")
        f.write(col_idx_np.tobytes())

        # weights (E, float32)
        weights_np = net.weights.numpy().astype("float32")
        f.write(weights_np.tobytes())

        # delay_ticks (E, int32)
        delays_np = net.delay_ticks.numpy().astype("int32")
        f.write(delays_np.tobytes())

        # base_omegas (N, float32)
        omegas_np = net.base_omegas.numpy().astype("float32")
        f.write(omegas_np.tobytes())

    print(f"[export] Done. {out_path} ({N} nodes, {E} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=200, help="Number of nodes (default: 200)"
    )
    parser.add_argument(
        "--out", type=str, default="leviathan_topology.bin", help="Output binary path"
    )
    args = parser.parse_args()
    export(args.n, args.out)
