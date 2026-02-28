#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// --- BIOLOGICAL CONFIGURATION ---
#define DT 0.01f
#define MAX_DELAY_TICKS 200 // corresponds to 2.0s max delay at DT=0.01

#define DEFAULT_INERTIA 1.0f
#define DEFAULT_DAMPING 0.5f

#define R_BASAL 0.1f
#define K_COST 0.05f
#define K_MAINT 0.02f
#define SYNAPTIC_W_MAX 5.0f

#define A_PLUS 0.1f
#define A_MINUS 0.12f
#define TAU_PLUS 0.05f
#define TAU_MINUS 0.05f

// --- NETWORK STATE ARRAYS (Device Pointers) ---
struct LeviathanDeviceState {
    int num_nodes;
    int current_tick;
    
    // Physics parameters
    float* d_base_omegas; // [N]
    
    // Core State Vectors [N]
    float* d_phi;
    float* d_phi_dot;
    float* d_atp;
    float* d_sensory_forcing; // S_i(t)
    
    // Neuromodulator levels (Could be scalars if global, or [N] if local diffusion)
    float DA;
    float NA;
    float SER;
    float ACH;

    // Topology: Compressed Sparse Row (CSR) format for weights
    int num_edges;
    int* d_csr_row_ptrs; // [N + 1]
    int* d_csr_col_indices; // [num_edges]
    float* d_csr_weights; // [num_edges]
    
    // Conduction Delays (aligned to sparse edges)
    // d_delay_ticks[edge_idx] = delay in ticks for this synapse
    int* d_delay_ticks; // [num_edges]
    
    // Ring Buffers for History [MAX_DELAY_TICKS * N]
    // Flattened 2D array: history[tick * N + node_id]
    float* d_history_phi;
    float* d_history_phi_dot;
    
    // Intermediate Array for the Synaptic Integration coupling sum
    float* d_coupling_sum; // [N]
};

// --- KERNEL DECLARATIONS ---

// Kernel 1: Updates S_i(t) based on external environment and evaluates Endocrine metrics
__global__ void kernel_sensory_endocrine(LeviathanDeviceState state);

// Kernel 2: Sparse matrix-vector multiply for \sum W_ij * sin(delayed_phi_j - \phi_i)
__global__ void kernel_synaptic_integration(LeviathanDeviceState state);

// Kernel 3: Solves the Second-Order DDE (RK2 or Verlet) to advance phi and phi_dot
__global__ void kernel_kinematics(LeviathanDeviceState state);

// Kernel 4: Evaluates PTDP causality on the delayed phase map and updates the CSR weights 
__global__ void kernel_plasticity_ptdp(LeviathanDeviceState state);

// Helper Kernel: Drain ATP and compute basic homeostatic scaling logic
__global__ void kernel_metabolism(LeviathanDeviceState state);

#endif
