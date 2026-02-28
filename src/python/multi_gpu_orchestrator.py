"""
multi_gpu.cu companion plan + Python orchestrator skeleton  (Phase 9I)

This file documents the multi-GPU partition strategy and provides a Python
wrapper that launches per-GPU processes. Full NCCL C++ skeleton is in
src/cuda/multi_gpu_nccl.cu.

Strategy:
  - Partition by block rows (BSR) across K GPUs
  - GPU k handles block rows [k*NB//K, (k+1)*NB//K)
  - Halo exchange: each GPU sends boundary block phases to neighbors via NCCL AllGather
  - Stream overlap: compute internal blocks while comm exchanges halo
"""

import os
import torch
import torch.multiprocessing as mp

DIST_AVAILABLE = False
try:
    import torch.distributed as dist

    DIST_AVAILABLE = dist.is_available()
except ImportError:
    pass

# NCCL is typically only available on Linux with CUDA
NCCL_AVAILABLE = (
    DIST_AVAILABLE
    and torch.cuda.is_available()
    and getattr(dist, "is_nccl_available", lambda: False)()
)


def _partition_csr(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    weights: torch.Tensor,
    delay_ticks: torch.Tensor,
    rank: int,
    world_size: int,
):
    """
    Partition CSR rows evenly across `world_size` GPUs.
    GPU `rank` gets rows [start, end).

    Returns local CSR slice + a list of remote_col_idx (halo dependencies).
    """
    N = row_ptr.shape[0] - 1
    start = rank * N // world_size
    end = (rank + 1) * N // world_size

    local_row_ptr = row_ptr[start : end + 1] - row_ptr[start]
    e_start = int(row_ptr[start])
    e_end = int(row_ptr[end])

    local_col_idx = col_idx[e_start:e_end]
    local_weights = weights[e_start:e_end]
    local_delays = delay_ticks[e_start:e_end]

    # Identify halo: columns that belong to remote GPUs
    remote_mask = (local_col_idx < start) | (local_col_idx >= end)
    remote_srcs = local_col_idx[remote_mask].unique()

    return {
        "row_ptr": local_row_ptr,
        "col_idx": local_col_idx,
        "weights": local_weights,
        "delays": local_delays,
        "node_range": (start, end),
        "remote_srcs": remote_srcs,
    }


def _worker(rank: int, world_size: int, shared_data: dict, result_queue):
    """
    Per-GPU simulation worker.
    Full implementation would:
      1. Move local partition to GPU rank
      2. Run kernel sequence for TICKS
      3. AllGather halo phases via NCCL after each tick
      4. Update halo buffer and continue
    """
    if world_size > 1 and DIST_AVAILABLE:
        # Default env for local rendezvous if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        backend = "nccl" if NCCL_AVAILABLE else "gloo"
        try:
            dist.init_process_group(backend, rank=rank, world_size=world_size)
            if NCCL_AVAILABLE:
                device = torch.device(f"cuda:{rank}")
            else:
                device = torch.device("cpu")
        except Exception as e:
            print(f"[multi_gpu] Rank {rank} failed to init {backend}: {e}")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    partition = _partition_csr(
        shared_data["row_ptr"],
        shared_data["col_idx"],
        shared_data["weights"],
        shared_data["delay_ticks"],
        rank,
        world_size,
    )

    start, end = partition["node_range"]
    N_local = end - start
    phase = torch.zeros(N_local, device=device)

    TICKS = shared_data.get("ticks", 100)

    for tick in range(TICKS):
        # ── COMPUTE internal (stub) ─────────────────────────────────────
        phase = phase + 0.01 * torch.randn_like(phase)

        # ── HALO EXCHANGE via AllGather ─────────────────────────────────
        if world_size > 1 and DIST_AVAILABLE and dist.is_initialized():
            all_phases = [torch.zeros_like(phase) for _ in range(world_size)]
            dist.all_gather(all_phases, phase)
            # Update halo buffer from all_phases here

    result_queue.put({"rank": rank, "final_phase_mean": float(phase.mean())})

    if DIST_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


def run_multi_gpu(
    row_ptr, col_idx, weights, delay_ticks, world_size: int = 2, ticks: int = 100
):
    """
    Launch `world_size` GPU workers.
    Returns list of {rank, final_phase_mean} dicts.
    """
    if world_size < 2 or not NCCL_AVAILABLE:
        print(
            "[multi_gpu] NCCL unavailable or world_size<2 — running single-process stub"
        )
        world_size = 1

    shared_data = {
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "weights": weights,
        "delay_ticks": delay_ticks,
        "ticks": ticks,
    }

    result_queue = mp.Queue()
    procs = []

    for rank in range(world_size):
        p = mp.Process(
            target=_worker, args=(rank, world_size, shared_data, result_queue)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    results = [result_queue.get() for _ in range(world_size)]
    return sorted(results, key=lambda r: r["rank"])


if __name__ == "__main__":
    # Minimal smoke test
    N, E = 64, 128
    row_ptr = torch.zeros(N + 1, dtype=torch.int32)
    col_idx = torch.randint(0, N, (E,), dtype=torch.int32)
    weights = torch.rand(E)
    delay_ticks = torch.ones(E, dtype=torch.int32) * 5

    for i in range(E):
        row_ptr[col_idx[i] + 1] += 1
    row_ptr = torch.cumsum(row_ptr, 0)

    results = run_multi_gpu(
        row_ptr, col_idx, weights, delay_ticks, world_size=1, ticks=10
    )
    print("Multi-GPU stub result:", results)
