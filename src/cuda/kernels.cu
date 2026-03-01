/*
 * kernels.cu — Phase 9: CUDA Feature Parity
 *
 * Upgrades over Phase 7 stub:
 *   - kernel_sensory_endocrine: reads d_sensory_input (host-writeable), updates
 * DA/NA/SER/ACH
 *   - kernel_kinematics: RK2 integrator, real NA noise (XORSHIFT per-thread)
 *   - kernel_plasticity_ptdp: atan2 phase wrapping (matches Python)
 *   - kernel_metabolism: per-node ATP drain + pruning flag
 *   - All cudaDeviceSynchronize() moved to host loop (one per tick)
 */
#include "kernels.h"
#include <math_constants.h>

/* ── Deterministic per-thread XORSHIFT RNG ─────────────────────────────── */
__device__ __forceinline__ float xorshift_uniform(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (state & 0xFFFFu) / 65535.0f - 0.5f; /* range [-0.5, 0.5] */
}

/* ─────────────────────────────────────────────────────────────────────────
 * 1. Sensory forcing + Endocrine update
 *    - Reads d_sensory_input written by host each tick from environment.
 *    - Thread 0 (block 0) computes mean_atp then updates DA/NA/SER/ACH.
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void kernel_sensory_endocrine(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /* Copy host-provided sensory signal into device array */
  if (i < state.num_nodes) {
    state.d_sensory_forcing[i] = state.d_sensory_input[i];
  }

  __syncthreads();

  /* Thread 0 of block 0: scalar endocrine update */
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    float mean_atp = 0.0f;
    for (int k = 0; k < state.num_nodes; k++)
      mean_atp += state.d_atp[k];
    mean_atp /= (float)state.num_nodes;

    float starving = 0.0f;
    for (int k = 0; k < state.num_nodes; k++)
      if (state.d_atp[k] <= ATP_PRUNE_THRESHOLD)
        starving += 1.0f;
    float error = starving / (float)state.num_nodes;

    float mean_vel = 0.0f;
    for (int k = 0; k < state.num_nodes; k++)
      mean_vel += fabsf(state.d_phi_dot[k]);
    mean_vel /= (float)state.num_nodes;

    /* DA: goal alignment proxy */
    float da_drive = ALPHA_DA * fmaxf(0.0f, mean_atp - ATP_TARGET);
    state.DA += (-state.DA / TAU_DA + da_drive) * DT;

    /* NA: threat / starvation */
    state.NA += (-(state.NA - NA_BASAL) / TAU_NA + ALPHA_NA * error) * DT;

    /* SER: ATP health */
    state.SER += (-(state.SER - 1.0f) / TAU_SER +
                  ALPHA_SER * (mean_atp - ATP_TARGET) / ATP_TARGET) *
                 DT;

    /* ACH: velocity novelty */
    float ach_drive = ALPHA_ACH * fmaxf(0.0f, mean_vel - THETA_NOVEL);
    state.ACH += (-(state.ACH - 1.0f) / TAU_ACH + ach_drive) * DT;

    /* Biological clamps */
    state.DA = fmaxf(0.0f, fminf(state.DA, 10.0f));
    state.NA = fmaxf(0.0f, fminf(state.NA, 5.0f));
    state.SER = fmaxf(0.1f, fminf(state.SER, 5.0f));
    state.ACH = fmaxf(0.1f, fminf(state.ACH, 5.0f));
  }
}

/* ─────────────────────────────────────────────────────────────────────────
 * 2. Synaptic integration (CSR sparse, O(E))
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void kernel_synaptic_integration(LeviathanDeviceState state) {
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= state.num_nodes)
    return;

  float current_phi = state.d_phi[dst];
  float coupling = 0.0f;

  int row_start = state.d_csr_row_ptrs[dst];
  int row_end = state.d_csr_row_ptrs[dst + 1];

  for (int e = row_start; e < row_end; e++) {
    int src = state.d_csr_col_indices[e];
    float w = state.d_csr_weights[e];
    int delay = state.d_delay_ticks[e];

    /* Source-major layout: history[src * MAX_DELAY_TICKS + slot] */
    int slot =
        ((state.current_tick - delay) % MAX_DELAY_TICKS + MAX_DELAY_TICKS) %
        MAX_DELAY_TICKS;
    float delayed_phi = state.d_history_phi[src * MAX_DELAY_TICKS + slot];

    coupling += w * __sinf(delayed_phi - current_phi);
  }

  state.d_coupling_sum[dst] = coupling * state.SER;
}

/* ─────────────────────────────────────────────────────────────────────────
 * 3. Kinematics — RK2 integrator + XORSHIFT noise + velocity clamp
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void kernel_kinematics(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  float phi = state.d_phi[i];
  float phi_dot = state.d_phi_dot[i];
  float omega = state.d_base_omegas[i];

  float coupling = state.d_coupling_sum[i];
  float s_force = state.d_sensory_forcing[i];

  /* XORSHIFT noise seeded per thread */
  uint32_t rng = state.d_rng[i];
  float noise = state.NA * xorshift_uniform(rng);
  state.d_rng[i] = rng;

  /* Active Inference: Update Error and Drive */
  float error = phi - state.d_phi_target[i];
  error = atan2f(__sinf(error), __cosf(error));
  state.d_error[i] = error;
  float ai_drive = -ALPHA_ACTIVE_INF * error;

  /* ACH reduces inertia (Cortex-only if mask is set) */
  float m_gain = state.ACH;
  if (state.d_cortex_mask && !state.d_cortex_mask[i]) {
    m_gain = 1.0f; // No gain outside cortex
  }
  float active_m = DEFAULT_INERTIA / fmaxf(0.1f, m_gain);

  /* RK2 midpoint */
  float a1 = (coupling + s_force + noise + ai_drive -
              DEFAULT_DAMPING * (phi_dot - omega)) /
             active_m;

  float phi_dot_mid = phi_dot + 0.5f * DT * a1;
  float phi_mid = phi + 0.5f * DT * phi_dot;

  float a2 = (coupling + s_force + noise + ai_drive -
              DEFAULT_DAMPING * (phi_dot_mid - omega)) /
             active_m;

  phi_dot += DT * a2;
  phi += DT * phi_dot_mid;

  /* Velocity clamp */
  phi_dot = fmaxf(-V_MAX, fminf(phi_dot, V_MAX));

  /* Phase wrap */
  phi = fmodf(phi, 2.0f * CUDART_PI_F);
  if (phi < 0.0f)
    phi += 2.0f * CUDART_PI_F;

  state.d_phi[i] = phi;
  state.d_phi_dot[i] = phi_dot;

  /* Update ring buffer (source-major layout) */
  int next_slot = (state.current_tick + 1) % MAX_DELAY_TICKS;
  state.d_history_phi[i * MAX_DELAY_TICKS + next_slot] = phi;
  state.d_history_phi_dot[i * MAX_DELAY_TICKS + next_slot] = phi_dot;
}

/* ─────────────────────────────────────────────────────────────────────────
 * 4. PTDP plasticity — atan2 wrapping (matches Python implementation)
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void kernel_plasticity_ptdp(LeviathanDeviceState state) {
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= state.num_nodes)
    return;

  float current_phi = state.d_phi[dst];
  float w_sum = 0.0f;

  int row_start = state.d_csr_row_ptrs[dst];
  int row_end = state.d_csr_row_ptrs[dst + 1];

  for (int e = row_start; e < row_end; e++) {
    int src = state.d_csr_col_indices[e];
    int delay = state.d_delay_ticks[e];
    float w = state.d_csr_weights[e];

    int slot =
        ((state.current_tick - delay) % MAX_DELAY_TICKS + MAX_DELAY_TICKS) %
        MAX_DELAY_TICKS;
    float delayed_phi = state.d_history_phi[src * MAX_DELAY_TICKS + slot];

    /* atan2 wrapping: maps delta to [-π, π] */
    float delta = delayed_phi - current_phi;
    delta = atan2f(__sinf(delta), __cosf(delta));

    float dw = (delta > 0.0f) ? state.DA * A_PLUS * expf(-delta / TAU_PLUS)
                              : -A_MINUS * expf(delta / TAU_MINUS);

    w += dw * DT;
    if (w < 0.0f)
      w = 0.0f;

    state.d_csr_weights[e] = w;
    w_sum += w;
  }

  /* Homeostatic normalization */
  if (w_sum > SYNAPTIC_W_MAX) {
    float scale = SYNAPTIC_W_MAX / w_sum;
    for (int e = row_start; e < row_end; e++)
      state.d_csr_weights[e] *= scale;
  }
}

/* ─────────────────────────────────────────────────────────────────────────
 * 5. Metabolism + ATP drain
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void kernel_metabolism(LeviathanDeviceState state) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= state.num_nodes)
    return;

  float phi_dot = fabsf(state.d_phi_dot[i]);
  float w_sum = 0.0f;

  int row_start = state.d_csr_row_ptrs[i];
  int row_end = state.d_csr_row_ptrs[i + 1];
  for (int e = row_start; e < row_end; e++)
    w_sum += state.d_csr_weights[e];

  float drain = (R_BASAL + K_COST * phi_dot + K_MAINT * w_sum) * DT;
  state.d_atp[i] -= drain;
}
