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

        # ── Spatial topology initialization ──
        # (This just sets up candidate regions; build_csr_topology will re-order nodes for locality)
        self.positions, self.base_omegas, self.region_labels = (
            generate_spatial_topology(num_nodes)
        )

        # ── Sparse CSR connectivity ────────────────────────────────────
        # Phase 11: build_csr_topology now returns:
        # row_ptr, col_idx, weights, delay_ticks, is_inhibitory, positions, region_labels
        (
            self.row_ptr,
            self.col_idx,
            self.weights,
            self.delay_ticks,
            self.is_inhibitory,
            self.positions,
            self.region_labels,
        ) = build_csr_topology(self.positions, self.region_labels)

        # ── Precomputed dst_idx for vectorized coupling ─────────────────
        # dst_idx[e] = destination node of edge e (expanded from row_ptr)
        self.dst_idx = self._build_dst_idx()

        # ── ATP Economy ────────────────────────────────────────────────
        self.atp = torch.ones(num_nodes) * INITIAL_ATP

        # ── Phase / velocity state ─────────────────────────────────────
        self.phi = torch.zeros(num_nodes)
        self.phi_dot = torch.zeros(num_nodes)

        # ── Active Inference (Generative Model) ───────────────────────
        self.phi_target = torch.zeros(num_nodes)  # Target/expected phase
        self.mu = torch.zeros(num_nodes)  # Internal belief
        self.error = torch.zeros(num_nodes)  # Prediction error
        self.ALPHA_ACTIVE_INF = 0.5  # Precision/gain on error

        # ── Phase 13: Biological Metrics ──────────────────────────────
        self.lyapunov_exponent = 0.0
        self.branching_ratio = 1.0
        self.prev_activation = torch.zeros(num_nodes)

        # Ring buffers  (MAX_DELAY_TICKS, N)
        self.history_phi = torch.zeros(MAX_DELAY_TICKS, num_nodes)
        self.history_phi_dot = torch.zeros(MAX_DELAY_TICKS, num_nodes)
        self.current_tick = 0

        # ── Physical parameters ────────────────────────────────────────
        self.m = torch.ones(num_nodes) * DEFAULT_INERTIA
        self.c = torch.ones(num_nodes) * DEFAULT_DAMPING

        # Cortex mask — ACh applied here only
        self._cortex_mask = torch.tensor(
            [label.startswith("cortex") for label in self.region_labels],
            dtype=torch.bool,
            device=self.phi.device,
        )

        # ── Neuromodulators ────────────────────────────────────────────
        self.DA = 1.0  # Dopamine     (reward gate / PTDP)
        self.NA = 0.1  # Noradrenaline (noise / arousal)
        self.SER = 1.0  # Serotonin    (coupling multiplier)
        self.ACH = 1.0  # Acetylcholine (inertia reduction — cortex)

        # ── Phase 14: Liquid State Machine Memory ─────────────────────
        from .memory import LiquidStateMachine

        self.lsm = LiquidStateMachine(input_dim=num_nodes, output_dim=2)
        self.lsm_loss = 0.0

        # ── Phase 16: Global Workspace (GWT) ──────────────────────────
        self.GWT_K = 10  # Capacity of consciousness (top-K nodes)
        self.GWT_GAIN = 0.5  # Strength of global broadcast
        self.workspace_indices = torch.zeros(self.GWT_K, dtype=torch.long)

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
        # Standard dynamics + Active Inference error minimization drive
        # dphi_dot += -alpha * (predicted - sensory)
        ai_drive = -self.ALPHA_ACTIVE_INF * self.error
        return (
            coupling + S + noise + ai_drive - self.c * (phi_dot - self.base_omegas)
        ) / inertia

    # ─────────────────────────────────────────────────────────────────────
    def step(self, external_sensory_input=None):
        """
        Advance the network by one timestep using RK2 (midpoint method).
        Returns: delayed_phi_by_edge — (E,) tensor used for PTDP.
        """
        N = self.num_nodes

        # ── Coupling (O(E)) ────────────────────────────────────────────
        coupling = self._sparse_coupling()

        # ── Global Workspace Broadcast (GWT) ──────────────────────────
        # Identify top-K most active nodes (highest phi_dot)
        # These nodes "broadcast" their state to the entire network
        _, self.workspace_indices = torch.topk(torch.abs(self.phi_dot), self.GWT_K)

        # Mean phase of the workspace nodes
        workspace_phi = torch.mean(self.phi[self.workspace_indices])

        # Broadcast term: global attractor towards workspace consensus
        gwt_broadcast = self.GWT_GAIN * torch.sin(workspace_phi - self.phi)

        # Add broadcast to coupling
        coupling += gwt_broadcast

        # ── Sensory + Noise ────────────────────────────────────────────
        S = (
            external_sensory_input
            if external_sensory_input is not None
            else torch.zeros(N)
        )
        noise = self.NA * torch.randn(N)

        # ── Active Inference: Update Error ────────────────────────────
        # Target = exponential moving average of sensory-driven state
        # (Simplified: filter sensory input into the target)
        target_alpha = 0.05
        self.phi_target = (
            1.0 - target_alpha
        ) * self.phi_target + target_alpha * external_sensory_input

        # Error = Actual - Expected (wrapped to [-pi, pi])
        self.error = torch.atan2(
            torch.sin(self.phi - self.phi_target), torch.cos(self.phi - self.phi_target)
        )

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

        # ── Phase 13: Biological Metrics ──────────────────────────────
        self._update_biological_metrics()

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
    def _update_biological_metrics(self):
        """
        Estimate Lyapunov exponent (stability) and Branching Ratio (criticality).
        """
        # 1. Lyapunov-ish: Track mean absolute change in velocity (divergence proxy)
        # In a real Lyapunov calculation, we'd perturb a shadow state.
        # Here we use a rolling estimate of velocity variance as a stability proxy.
        current_accel = torch.abs(
            self.phi_dot
            - self.history_phi_dot[(self.current_tick - 1) % MAX_DELAY_TICKS]
        )
        self.lyapunov_exponent = 0.95 * self.lyapunov_exponent + 0.05 * float(
            torch.mean(current_accel)
        )

        # 2. Branching Ratio: <Active Nodes at t> / <Active Nodes at t-1>
        # Define 'active' as a phase-velocity threshold.
        threshold = 0.1
        active_now = (torch.abs(self.phi_dot) > threshold).float()
        active_prev = (torch.abs(self.prev_activation) > threshold).float()

        counts_now = torch.sum(active_now)
        counts_prev = torch.sum(active_prev)

        if counts_prev > 0:
            ratio = float(counts_now / counts_prev)
            self.branching_ratio = 0.99 * self.branching_ratio + 0.01 * ratio

        self.prev_activation = self.phi_dot.clone()

    # ─────────────────────────────────────────────────────────────────────
    @property
    def cognitive_entropy(self) -> float:
        """
        Calculates Shannon Entropy of the network's phase-velocity distribution.
        Higher entropy indicates higher cognitive diversity/arousal.
        """
        # Normalize phi_dot into a probability distribution
        # We use softmax over abs(phi_dot) to get a distribution
        probs = torch.softmax(torch.abs(self.phi_dot) * 2.0, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        return float(entropy)

    # ─────────────────────────────────────────────────────────────────────
    def prune_weakest_starving(self):
        """
        For each node whose ATP < threshold, prune its weakest incoming synapse.
        """
        starving = (self.atp < ATP_PRUNE_THRESHOLD).nonzero(as_tuple=True)[0]
        for dst in starving:
            dst = int(dst)
            start = int(self.row_ptr[dst])
            end = int(self.row_ptr[dst + 1])
            if end == start:
                continue
            local_min = int(torch.argmin(self.weights[start:end]))
            self.weights[start + local_min] = 0.0

    # ─────────────────────────────────────────────────────────────────────
    def synaptic_birth(self, surplus_atp=2000.0, birth_rate=0.01):
        """
        Structural Plasticity: Grow new synapses for nodes with high ATP.
        """
        # Find nodes with high ATP surplus
        wealthy = (self.atp > surplus_atp).nonzero(as_tuple=True)[0]
        if len(wealthy) == 0:
            return

        added_edges = []
        for dst in wealthy:
            if torch.rand(1) > birth_rate:
                continue

            dst = int(dst)
            # Find a spatial neighbor (simplified: random node for now,
            # ideally closer in self.positions)
            src = int(torch.randint(0, self.num_nodes, (1,)))
            if src == dst:
                continue

            # Create new edge
            # Initial weight is very small/exploratory
            is_inh = self.is_inhibitory[src]
            w = -0.01 if is_inh else 0.01

            # Distance-based delay
            dist = torch.norm(self.positions[src] - self.positions[dst])
            delay = int(dist / 5.0) + 1  # 5 units per tick
            delay = min(delay, MAX_DELAY_TICKS - 1)

            added_edges.append((dst, src, w, delay))

        if not added_edges:
            return

        # CSR Re-allocation logic (Batch update)
        # Note: This is an expensive operation in Python/Torch.
        # Ideally we'd pre-allocate 'ghost' edges.
        for dst, src, w, delay in added_edges:
            # Shift row_ptr
            self.row_ptr[dst + 1 :] += 1

            # Insert into col_idx, weights, delay_ticks, is_inhibitory
            insert_pos = int(self.row_ptr[dst + 1]) - 1

            self.col_idx = torch.cat(
                [
                    self.col_idx[:insert_pos],
                    torch.tensor([src]),
                    self.col_idx[insert_pos:],
                ]
            )
            self.weights = torch.cat(
                [
                    self.weights[:insert_pos],
                    torch.tensor([w]),
                    self.weights[insert_pos:],
                ]
            )
            self.delay_ticks = torch.cat(
                [
                    self.delay_ticks[:insert_pos],
                    torch.tensor([delay]),
                    self.delay_ticks[insert_pos:],
                ]
            )

        # Update dst_idx
        self.dst_idx = self._build_dst_idx()
