import torch
import math
from .config import *
from .topology import generate_spatial_topology, calculate_conduction_delays, initialize_adjacency_matrix

class LeviathanNetwork:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        
        # 1. Physics Properties
        self.positions, self.base_omegas, self.region_labels = generate_spatial_topology(num_nodes)
        
        # Delay Matrix (N, N) - seconds
        self.delay_sec = calculate_conduction_delays(self.positions, CONDUCTION_VELOCITY)
        # Delay Matrix (N, N) - ticks for ring buffer lookup
        self.delay_ticks = (self.delay_sec / DT).to(torch.long)
        # Cap to max
        self.delay_ticks = torch.clamp(self.delay_ticks, min=0, max=MAX_DELAY_TICKS - 1)
        
        # Initial Sparse Weights (N, N)
        self.weights = initialize_adjacency_matrix(self.positions, lambda_decay=TOPOLOGY_LAMBDA)
        
        # ATP Economy (N,)
        self.atp = torch.ones(num_nodes) * INITIAL_ATP
        
        # 2. State & Ring Buffers
        self.phi = torch.zeros(num_nodes)          # Phase angle
        self.phi_dot = torch.zeros(num_nodes)      # Phase velocity
        
        # History is (MaxTicks, N) for fast slice fetching
        self.history_phi = torch.zeros((MAX_DELAY_TICKS, num_nodes))
        self.history_phi_dot = torch.zeros((MAX_DELAY_TICKS, num_nodes))
        self.current_tick = 0
        
        # 3. Local Parameters
        self.m = torch.ones(num_nodes) * DEFAULT_INERTIA
        self.c = torch.ones(num_nodes) * DEFAULT_DAMPING
        
        # 4. Neuromodulators
        self.DA = 1.0     # Dopamine
        self.NA = 0.1     # Noradrenaline (Noise)
        self.SER = 1.0    # Serotonin (Coupling mult)
        self.ACH = 1.0    # Acetylcholine (Inertia scale)
        
    def step(self, external_sensory_input=None):
        """ Integrates the Second-Order DDEs for 1 step (Verlet or RK2) """
        N = self.num_nodes
        
        # Pre-calculate indices we need to fetch from history
        # target_history_idx[i, j] = the absolute tick index in the ring buffer 
        # that represents phi_j(t - tau_ij)
        
        # current absolute tick% MAX
        curr_idx = self.current_tick % MAX_DELAY_TICKS
        
        # target_idx = (curr - delay) % MAX
        # shape: (N_dst, N_source)
        # However, to index the history cleanly:
        # history_phi shape is (MAX_TICKS, N_nodes)
        
        # We need a matrix of delayed phases: delayed_phi[i, j] = phi_j(t - tau_ij)
        # To vectorize this simply, we can use torch.gather or advanced indexing.
        
        # For small N in python, explicit loop or gather:
        # Flat history: history_phi(T, N)
        
        delayed_phi = torch.zeros((N, N))
        for dst in range(N):
            # The delays from all sources to this destination 
            # self.delay_ticks[dst, :] -> (N,)
            offsets = (curr_idx - self.delay_ticks[dst, :]) % MAX_DELAY_TICKS
            
            # Retrieve from history for all sources j
            # We want history_phi[ offsets[j], j ]
            source_indices = torch.arange(N)
            delayed_phi[dst, :] = self.history_phi[offsets, source_indices]
            
        # --- Synaptic Integration ---
        # Sum_j W_ij * sin(delayed_phi_j - current_phi_i)
        
        # shape (N, N): each row dst is current_phi_dst
        current_phi_matrix = self.phi.unsqueeze(1).expand(N, N) 
        phase_diff = delayed_phi - current_phi_matrix 
        
        # Coupling term
        coupling = self.SER * torch.sum(self.weights * torch.sin(phase_diff), dim=1)
        
        # --- Endocrine & Sensory ---
        S = external_sensory_input if external_sensory_input is not None else 0.0
        noise = self.NA * torch.randn(N)
        
        active_inertia = self.m / self.ACH  # ACH decreases inertia in cortex conceptually
        # (For simplicity we apply it globally here, ideally restricted to cortex per spec)
        
        # --- Second Order Verlet/Euler ---
        # m*phi_ddot + c*(phi_dot - omega) = coupling + S + Noise
        
        accel = (coupling + S + noise - self.c * (self.phi_dot - self.base_omegas)) / active_inertia
        
        # Update velocities
        self.phi_dot = self.phi_dot + accel * DT
        
        # Update phases
        self.phi = self.phi + self.phi_dot * DT
        
        # Normalize phases
        self.phi = self.phi % (2.0 * math.pi)
        
        # --- Save to History ---
        next_idx = (self.current_tick + 1) % MAX_DELAY_TICKS
        self.history_phi[next_idx, :] = self.phi
        self.history_phi_dot[next_idx, :] = self.phi_dot
        self.current_tick += 1
        
        return delayed_phi # returned for plasticity computation
