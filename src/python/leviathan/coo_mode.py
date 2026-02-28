"""
coo_mode.py — Phase 10: COO (Coordinate List) memory-streaming graph

Simpler than CSR for very large edge sets: stores (src, dst, weights) as flat arrays.
Coupling computed via scatter_add in chunks → avoids large intermediate tensors.
Preferred over CSR when edge sets don't fit in VRAM at once.
"""

import torch


class COOGraph:
    """
    Memory-bandwidth friendly COO graph.
    Processes edges in chunks to cap peak memory usage.
    """

    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor,
        chunk_size: int = 500_000,
    ):
        self.src = src  # (E,) long
        self.dst = dst  # (E,) long
        self.weights = weights  # (E,) float
        self.chunk_size = chunk_size
        self.N = int(max(dst.max(), src.max())) + 1

    def coupling(
        self,
        phi: torch.Tensor,
        history_phi: torch.Tensor,
        delays: torch.Tensor,
        hist_idx: int,
        MAX_DELAY: int,
    ) -> torch.Tensor:
        """
        Vectorized coupling with chunked streaming.
        Feeds edges through in `chunk_size` slices to cap memory.
        """
        coupling = torch.zeros(self.N, device=phi.device)

        for i in range(0, len(self.src), self.chunk_size):
            s = self.src[i : i + self.chunk_size]
            d = self.dst[i : i + self.chunk_size]
            w = self.weights[i : i + self.chunk_size]
            dl = delays[i : i + self.chunk_size]

            slots = (hist_idx - dl) % MAX_DELAY
            phi_del = history_phi[slots, s]
            phi_dst = phi[d]

            coupling.scatter_add_(0, d, w * torch.sin(phi_del - phi_dst))

        return coupling

    @classmethod
    def from_csr(
        cls, row_ptr, col_idx, weights, delay_ticks, chunk_size: int = 500_000
    ) -> "COOGraph":
        """Convert CSR to COO using dst_idx expansion."""
        N = row_ptr.shape[0] - 1
        E = col_idx.shape[0]
        dst = torch.zeros(E, dtype=torch.long)
        for i in range(N):
            s, e = int(row_ptr[i]), int(row_ptr[i + 1])
            dst[s:e] = i
        return cls(col_idx.long(), dst, weights, chunk_size)
