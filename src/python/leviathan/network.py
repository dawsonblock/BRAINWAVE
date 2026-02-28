"""
network.py — Sparse CSR Leviathan DDE Engine (Phase 9)

Key upgrades over Phase 7:
  - O(E) sparse CSR coupling (was O(N²) dense loop)
  - Per-edge delay lookups via indexed gather — no NxN delayed_phi matrix
  - RK2 integrator (was Euler — numerically stabler)
  - Region-scoped ACh (cortex only, per spec)
  - Velocity clamping + Lyapunov energy monitor
  - Pruning on starvation (weakest incoming synapse removed per row)
"""

import torch
import math
from .config import (
    DT,
    MAX_DELAY_TICKS,
    INITIAL_ATP,
    DEFAULT_INERTIA,
    DEFAULT_DAMPING,
    ATP_DEATH_THRESHOLD,
    SYNAPTIC_W_MAX,
)
from .topology import generate_spatial_topology, build_csr_topology

# Alias matching spec language
ATP_PRUNE_THRESHOLD = ATP_DEATH_THRESHOLD


V_MAX = 50.0  # rad/s velocity clamp
LYAPUNOV_WARN = 1e6  # energy threshold before logging divergence warning
CORTEX_LABEL = "cortex"


class LeviathanNetwork:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        # ── Spatial topology ──────────────────────────────────────────
        self.positions, self.base_omegas, self.region_labels = (
            generate_spatial_topology(num_nodes)
        )

        # ── Sparse CSR connectivity ────────────────────────────────────
        # row_ptr: (N+1,)  — row i spans edges [row_ptr[i], row_ptr[i+1])
        # col_idx: (E,)    — source node for each edge (row = destination)
        # weights: (E,)    — synaptic strength
        # delay_ticks: (E,) — integer ring-buffer offset
        (self.row_ptr, self.col_idx, self.weights, self.delay_ticks) = (
            build_csr_topology(self.positions, self.region_labels)
        )

        # ── Precomputed dst_idx for vectorized coupling ─────────────────
        # dst_idx[e] = destination node of edge e (expanded from row_ptr)
        self.dst_idx = self._build_dst_idx()

        # ── ATP Economy ────────────────────────────────────────────────
        self.atp = torch.ones(num_nodes) * INITIAL_ATP

        # ── Phase / velocity state ─────────────────────────────────────
        self.phi = torch.zeros(num_nodes)
        self.phi_dot = torch.zeros(num_nodes)

        # Ring buffers  (MAX_DELAY_TICKS, N)
        self.history_phi = torch.zeros(MAX_DELAY_TICKS, num_nodes)
        self.history_phi_dot = torch.zeros(MAX_DELAY_TICKS, num_nodes)
        self.current_tick = 0

        # ── Physical parameters ────────────────────────────────────────
        self.m = torch.ones(num_nodes) * DEFAULT_INERTIA
        self.c = torch.ones(num_nodes) * DEFAULT_DAMPING

        # Cortex mask — ACh applied here only
        self._cortex_mask = torch.tensor(
            [label == CORTEX_LABEL for label in self.region_labels], dtype=torch.bool
        )

        # ── Neuromodulators ────────────────────────────────────────────
        self.DA = 1.0  # Dopamine     (reward gate / PTDP)
        self.NA = 0.1  # Noradrenaline (noise / arousal)
        self.SER = 1.0  # Serotonin    (coupling multiplier)
        self.ACH = 1.0  # Acetylcholine (inertia reduction — cortex)

    # ─────────────────────────────────────────────────────────────────────
    def _build_dst_idx(self) -> torch.Tensor:
        """Expand row_ptr into per-edge destination indices (precomputed once)."""
        N, E = self.num_nodes, self.col_idx.shape[0]
        dst = torch.zeros(E, dtype=torch.long)
        for i in range(N):
            s, e = int(self.row_ptr[i]), int(self.row_ptr[i + 1])
            dst[s:e] = i
        return dst

    # ─────────────────────────────────────────────────────────────────────
    def _sparse_coupling(self) -> torch.Tensor:
        """
        Fully vectorized sparse coupling via scatter_add — no Python loop.
        Complexity: O(E)  — single pass over all edges.
        """
        hist = self.current_tick % MAX_DELAY_TICKS
        srcs = self.col_idx.long()  # (E,) source nodes
        dsts = self.dst_idx  # (E,) destination nodes
        delays = self.delay_ticks.long()  # (E,) per-edge delay

        slots = (hist - delays) % MAX_DELAY_TICKS  # (E,)
        phi_del = self.history_phi[slots, srcs]  # (E,)
        phi_dst = self.phi[dsts]  # (E,)

        edge_contrib = self.weights * torch.sin(phi_del - phi_dst)  # (E,)

        coupling = torch.zeros(self.num_nodes)
        coupling.scatter_add_(0, dsts, edge_contrib)  # O(E)

        return self.SER * coupling

    # ─────────────────────────────────────────────────────────────────────
    def _accel(
        self,
        phi: torch.Tensor,
        phi_dot: torch.Tensor,
        coupling: torch.Tensor,
        S: torch.Tensor,
        noise: torch.Tensor,
        inertia: torch.Tensor,
    ) -> torch.Tensor:
        """RHS of the second-order DDE: acceleration computation."""
        return (coupling + S + noise - self.c * (phi_dot - self.base_omegas)) / inertia

    # ─────────────────────────────────────────────────────────────────────
    def step(self, external_sensory_input=None):
        """
        Advance the network by one timestep using RK2 (midpoint method).
        Returns: delayed_phi_by_edge — (E,) tensor used for PTDP.
        """
        N = self.num_nodes

        # ── Coupling (O(E)) ────────────────────────────────────────────
        coupling = self._sparse_coupling()

        # ── Sensory + Noise ────────────────────────────────────────────
        S = (
            external_sensory_input
            if external_sensory_input is not None
            else torch.zeros(N)
        )
        noise = self.NA * torch.randn(N)

        # ── Region-scoped ACh: inertia reduction in cortex only ────────
        inertia = self.m.clone()
        inertia[self._cortex_mask] = self.m[self._cortex_mask] / max(self.ACH, 0.1)

        # ── RK2 (midpoint) ─────────────────────────────────────────────
        a1 = self._accel(self.phi, self.phi_dot, coupling, S, noise, inertia)

        phi_mid = self.phi + 0.5 * DT * self.phi_dot
        phi_dot_mid = self.phi_dot + 0.5 * DT * a1

        a2 = self._accel(phi_mid, phi_dot_mid, coupling, S, noise, inertia)

        self.phi_dot = self.phi_dot + DT * a2
        self.phi = self.phi + DT * phi_dot_mid

        # ── Stability clamps ───────────────────────────────────────────
        self.phi_dot = self.phi_dot.clamp(-V_MAX, V_MAX)
        self.phi = self.phi % (2.0 * math.pi)

        # Lyapunov monitor
        energy = float(torch.mean(self.phi_dot**2 + self.phi**2))
        if energy > LYAPUNOV_WARN:
            print(f"[WARN] Lyapunov energy={energy:.2e} at tick {self.current_tick}")

        # ── Update ring buffer ─────────────────────────────────────────
        next_idx = (self.current_tick + 1) % MAX_DELAY_TICKS
        self.history_phi[next_idx] = self.phi
        self.history_phi_dot[next_idx] = self.phi_dot
        self.current_tick += 1

        # Return per-edge delayed phases for PTDP (avoids recomputing)
        return self._get_delayed_phi_per_edge()

    # ─────────────────────────────────────────────────────────────────────
    def _get_delayed_phi_per_edge(self) -> torch.Tensor:
        """
        Return delayed source phase for each edge — (E,) tensor.
        Used by plasticity.py without re-allocating NxN.
        """
        hist = self.current_tick % MAX_DELAY_TICKS
        slots = (hist - self.delay_ticks) % MAX_DELAY_TICKS
        return self.history_phi[slots, self.col_idx]

    # ─────────────────────────────────────────────────────────────────────
    def prune_weakest_starving(self):
        """
        For each node whose ATP < threshold, prune its weakest incoming synapse.
        Called from endocrine.update_metabolism.
        """
        N = self.num_nodes
        starving = (self.atp < ATP_PRUNE_THRESHOLD).nonzero(as_tuple=True)[0]
        for dst in starving:
            dst = int(dst)
            start = int(self.row_ptr[dst])
            end = int(self.row_ptr[dst + 1])
            if end == start:
                continue
            local_min = int(torch.argmin(self.weights[start:end]))
            self.weights[start + local_min] = 0.0
