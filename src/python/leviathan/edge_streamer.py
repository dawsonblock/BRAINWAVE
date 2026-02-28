"""
edge_streamer.py — Phase 10: Edge streaming for 1M+ edge graphs

Wraps COO graph with a streaming iterator that yields chunks.
Used when even the full COO indices don't fit in GPU memory at once.
"""

import torch
from typing import Iterator, Tuple


class EdgeStreamer:
    """
    Process edges in chunks to avoid memory spikes on large graphs.
    Each chunk is a (src, dst, weights, delays) tuple sized <= chunk_size.
    """

    def __init__(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor,
        delays: torch.Tensor,
        chunk_size: int = 500_000,
    ):
        self.src = src
        self.dst = dst
        self.weights = weights
        self.delays = delays
        self.chunk_size = chunk_size

    def chunks(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        E = len(self.src)
        for i in range(0, E, self.chunk_size):
            yield (
                self.src[i : i + self.chunk_size],
                self.dst[i : i + self.chunk_size],
                self.weights[i : i + self.chunk_size],
                self.delays[i : i + self.chunk_size],
            )

    def coupling(
        self,
        phi: torch.Tensor,
        history_phi: torch.Tensor,
        hist_idx: int,
        MAX_DELAY: int,
    ) -> torch.Tensor:
        """Single-pass streaming coupling."""
        N = phi.shape[0]
        coupling = torch.zeros(N, device=phi.device)

        for s, d, w, dl in self.chunks():
            slots = (hist_idx - dl.long()) % MAX_DELAY
            phi_del = history_phi[slots, s.long()]
            coupling.scatter_add_(0, d.long(), w * torch.sin(phi_del - phi[d.long()]))

        return coupling
