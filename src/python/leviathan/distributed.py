"""
distributed.py — Phase 10: torch.distributed / NCCL wrapper

Thin wrapper for NCCL AllReduce — used by multi-GPU coupling aggregation.
Falls back gracefully on single-GPU or CPU runs.
"""

import torch

_DIST_AVAILABLE = False

try:
    import torch.distributed as dist

    _DIST_AVAILABLE = True
except ImportError:
    pass


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> bool:
    """Initialize process group. Returns True if successful."""
    if not _DIST_AVAILABLE:
        return False
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return True
    except Exception as e:
        print(f"[distributed] init failed: {e}")
        return False


def allreduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place AllReduce (sum) if distributed, else no-op."""
    if _DIST_AVAILABLE and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def is_primary() -> bool:
    """True if this is rank 0 (or not distributed)."""
    if _DIST_AVAILABLE and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def cleanup():
    if _DIST_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()
