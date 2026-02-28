#include "kernels.h"
#include <cmath>
#include <math_functions.h>

__global__ void kernel_sensory_endocrine(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  // In Leviathan v2.0, the sensory environment dictates S_i(t)
  // S_i(t) = I_0 / d * sin(theta - phi_i)
  // This is typically written by host memory async, here we stub with 0
  state.d_sensory_forcing[i] = 0.0f;

  // Only Thread 0 runs the global Endocrine Integrations
  if (i == 0) {
    // Not 100% parallel friendly to do it here, but convenient since it's
    // scalar state (DA, NA, SER, ACH) would be updated here.
  }
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
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= state.num_nodes)
    return;

  float current_phi = state.d_phi[dst];
  float w_sum = 0.0f;

  int row_start = state.d_csr_row_ptrs[dst];
  int row_end = state.d_csr_row_ptrs[dst + 1];

  // Pass 1: Apply PTDP updates
  for (int edge = row_start; edge < row_end; edge++) {
    int src = state.d_csr_col_indices[edge];
    float w = state.d_csr_weights[edge];
    int delay = state.d_delay_ticks[edge];

    int target_tick = (state.current_tick - delay) % MAX_DELAY_TICKS;
    if (target_tick < 0) {
      target_tick += MAX_DELAY_TICKS;
    }

    float delayed_phi =
        state.d_history_phi[target_tick * state.num_nodes + src];
    float delta_phi = delayed_phi - current_phi; // phi_j(t - \tau) - phi_i(t)

    // Normalize delta phase to [-PI, PI]
    delta_phi = fmodf(delta_phi + M_PI, 2.0f * M_PI);
    if (delta_phi < 0)
      delta_phi += 2.0f * M_PI;
    delta_phi -= M_PI;

    float dw = 0.0f;
    if (delta_phi > 0.0f) {
      // Node j leads Node i
      dw = state.DA * A_PLUS * expf(-delta_phi / TAU_PLUS);
    } else {
      // Node i leads Node j
      dw = -A_MINUS * expf(delta_phi / TAU_MINUS);
    }

    w += dw * DT;
    if (w < 0.0f)
      w = 0.0f; // Weights cannot be negative

    state.d_csr_weights[edge] = w;
    w_sum += w;
  }

  // Pass 2: Homeostatic Normalization
  if (w_sum > SYNAPTIC_W_MAX) {
    float scale = SYNAPTIC_W_MAX / w_sum;
    for (int edge = row_start; edge < row_end; edge++) {
      state.d_csr_weights[edge] *= scale;
    }
  }
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
