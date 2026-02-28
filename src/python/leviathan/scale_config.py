"""
scale_config.py — Phase 10: Configuration for ultra-large runs
"""


class ScaleConfig:
    """Runtime configuration for high-N simulations."""

    def __init__(
        self,
        max_delay: int = 8,
        use_half_precision: bool = True,
        edge_chunk_size: int = 500_000,
        device: str = "cuda",
        ptdp_interval: int = 10,  # apply plasticity every N ticks (CPU cost reduction)
    ):
        self.max_delay = max_delay
        self.use_half_precision = use_half_precision
        self.edge_chunk_size = edge_chunk_size
        self.device = device
        self.ptdp_interval = ptdp_interval


class MemoryBudget:
    """Estimates memory usage for a given N/E configuration."""

    @staticmethod
    def estimate(N: int, E: int, max_delay: int = 16, fp16: bool = False) -> dict:
        bytes_per = 2 if fp16 else 4
        phi_state = N * 2 * bytes_per  # phi + phi_dot
        atp = N * bytes_per
        history = N * max_delay * bytes_per
        edges = E * 3 * bytes_per  # weights + delays (int) + col_idx (int)
        total = phi_state + atp + history + edges

        return {
            "N": N,
            "E": E,
            "phi_state_MB": phi_state / 1e6,
            "history_MB": history / 1e6,
            "edges_MB": edges / 1e6,
            "total_MB": total / 1e6,
            "fits_8GB_GPU": total < 8e9,
            "fits_24GB_GPU": total < 24e9,
        }


if __name__ == "__main__":
    for N, E in [(10_000, 500_000), (100_000, 5_000_000), (500_000, 20_000_000)]:
        b = MemoryBudget.estimate(N, E)
        print(
            f"N={N:>7}  E={E:>10}  total={b['total_MB']:.0f}MB  "
            f"8GB={'✓' if b['fits_8GB_GPU'] else '✗'}  "
            f"24GB={'✓' if b['fits_24GB_GPU'] else '✗'}"
        )
