"""
partition.py — Phase 10: Multi-GPU edge sharding

Partitions CSR edges across K GPUs. Each GPU owns a full copy of node state
but processes only its assigned edge slice for coupling computation.
Coupling contributions are summed across GPUs via AllReduce.
"""

import torch
from typing import List


class GraphPartition:
    """
    Edge-based partitioning across multiple devices.
    Each partition owns a subset of edges but full node state.
    """

    def __init__(self, network, devices: List[str]):
        self.devices = [torch.device(d) for d in devices]
        self.partitions = []

        E = int(network.col_idx.shape[0])
        edges_per = E // len(self.devices)

        for i, dev in enumerate(self.devices):
            start = i * edges_per
            end = (i + 1) * edges_per if i < len(self.devices) - 1 else E

            part = {
                "col_idx": network.col_idx[start:end].long().to(dev),
                "dst_idx": network.dst_idx[start:end].long().to(dev),
                "weights": network.weights[start:end].to(dev),
                "delay": network.delay_ticks[start:end].long().to(dev),
                "device": dev,
            }
            self.partitions.append(part)

    def gather_coupling(
        self,
        phi: torch.Tensor,
        history_phi: torch.Tensor,
        hist_idx: int,
        MAX_DELAY: int,
    ) -> torch.Tensor:
        """
        Compute coupling from all partitions and sum on CPU.
        Each GPU computes its edge slice, result transferred back.
        """
        N = phi.shape[0]
        total_coup = torch.zeros(N)

        for part in self.partitions:
            dev = part["device"]
            phi_dev = phi.to(dev)
            hist_dev = history_phi.to(dev)

            slots = (hist_idx - part["delay"]) % MAX_DELAY
            phi_del = hist_dev[slots, part["col_idx"]]
            phi_dst = phi_dev[part["dst_idx"]]

            edge = part["weights"] * torch.sin(phi_del - phi_dst)

            coup = torch.zeros(N, device=dev)
            coup.scatter_add_(0, part["dst_idx"], edge)

            total_coup += coup.cpu()

        return total_coup
