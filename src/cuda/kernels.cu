#include "kernels.h"
#include <math_functions.h>

__global__ void kernel_sensory_endocrine(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  // In Leviathan v2.0, the sensory environment dictates S_i(t)
  // S_i(t) = I_0 / d * sin(theta - phi_i)
  // For now we assume S_i(t) is written to device memory asynchronously by a
  // host AI agent
}

__global__ void kernel_synaptic_integration(LeviathanDeviceState state) {
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= state.num_nodes)
    return;

  // Calculate sum W_ij * sin(delayed_phi_j - current_phi_dst)
  float coupling = 0.0f;
  float current_phi = state.d_phi[dst];

  int row_start = state.d_csr_row_ptrs[dst];
  int row_end = state.d_csr_row_ptrs[dst + 1];

  for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
    int src = state.d_csr_col_indices[edge_idx];
    float w = state.d_csr_weights[edge_idx];
    int delay = state.d_delay_ticks[edge_idx]; // delay = tau_ij / DT

    // Find history index: (current_tick - delay) % MAX_DELAY_TICKS
    int target_tick = (state.current_tick - delay) % MAX_DELAY_TICKS;
    if (target_tick < 0) {
      target_tick += MAX_DELAY_TICKS;
    }

    // Load delayed phase from ring buffer
    float delayed_phi =
        state.d_history_phi[target_tick * state.num_nodes + src];

    coupling += w * sinf(delayed_phi - current_phi);
  }

  state.d_coupling_sum[dst] =
      coupling * state.SER; // Serotonin scales coupling globally
}

__global__ void kernel_kinematics(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  float phi = state.d_phi[i];
  float phi_dot = state.d_phi_dot[i];
  float omega = state.d_base_omegas[i];

  // Calculate acceleration:
  // m*phi_ddot + c*(phi_dot - omega) = coupling + S
  float coupling = state.d_coupling_sum[i];
  float s_force = state.d_sensory_forcing[i];

  // Simplified noise for demo (in production use curand)
  float noise = state.NA * 0.0f;

  // Acetylcholine lowers inertia
  float active_m = DEFAULT_INERTIA / state.ACH;

  float accel =
      (coupling + s_force + noise - DEFAULT_DAMPING * (phi_dot - omega)) /
      active_m;

  // Verlet / Euler Step
  phi_dot += accel * DT;
  phi += phi_dot * DT;

  // Wrap to 2PI
  phi = fmodf(phi, 2.0f * M_PI);
  if (phi < 0.0f) {
    phi += 2.0f * M_PI;
  }

  // Save state back
  state.d_phi[i] = phi;
  state.d_phi_dot[i] = phi_dot;

  // Update history ring buffer for the *next* tick index
  int next_tick = (state.current_tick + 1) % MAX_DELAY_TICKS;
  state.d_history_phi[next_tick * state.num_nodes + i] = phi;
  state.d_history_phi_dot[next_tick * state.num_nodes + i] = phi_dot;
}

__global__ void kernel_plasticity_ptdp(LeviathanDeviceState state) {
  int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge_idx >= state.num_edges)
    return;

  // Recompute dst from binary search or store an array of dsts for edges.
  // For PTDP, we actually probably want threads over Nodes instead of Edges,
  // or we store an array of d_csr_dst_indices parallel to col_indices.
  // To simplify: let's do 1 thread per dst node, looping its edges.
}

__global__ void kernel_metabolism(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  float phi_dot = fabs(state.d_phi_dot[i]);

  // Cost of synapses
  float w_sum = 0.0f;
  int row_start = state.d_csr_row_ptrs[i];
  int row_end = state.d_csr_row_ptrs[i + 1];

  for (int edge = row_start; edge < row_end; edge++) {
    w_sum += state.d_csr_weights[edge];
  }

  float drain = (R_BASAL + K_COST * phi_dot + K_MAINT * w_sum) * DT;
  state.d_atp[i] -= drain;
}
