#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

/* ── Physical constants ─────────────────────────────────────────────────── */
#define DT 0.01f
#define MAX_DELAY_TICKS 200

#define DEFAULT_INERTIA 1.0f
#define DEFAULT_DAMPING 0.5f
#define V_MAX 50.0f /* rad/s velocity clamp (Phase 9) */

#define R_BASAL 0.1f
#define K_COST 0.05f
#define K_MAINT 0.02f
#define SYNAPTIC_W_MAX 10.0f
#define ATP_PRUNE_THRESHOLD 10.0f

/* Active Inference Drive */
#define ALPHA_ACTIVE_INF 0.5f

#define A_PLUS 0.1f
#define A_MINUS 0.12f
#define TAU_PLUS 0.5f
#define TAU_MINUS 0.5f

/* Endocrine constants (Phase 11 alignment) */
#define ATP_TARGET 800.0f
#define NA_BASAL 0.1f
#define THETA_NOVEL 5.0f

#define TAU_DA 5.0f
#define TAU_NA 3.0f
#define TAU_SER 10.0f
#define TAU_ACH 2.0f

#define ALPHA_DA 0.5f
#define ALPHA_NA 0.8f
#define ALPHA_SER 0.3f
#define ALPHA_ACH 0.6f

/* ── Device state structure ─────────────────────────────────────────────── */
struct LeviathanDeviceState {
  int num_nodes;
  int num_edges;
  int current_tick;

  /* Neuromodulators (global scalars) */
  float DA, NA, SER, ACH;

  /* Node parameters */
  float *d_base_omegas;  /* [N] */
  bool *d_is_inhibitory; /* [N] Dale's Principle */
  bool *d_cortex_mask;   /* [N] for area-scoped ACH */

  /* Core state */
  float *d_phi;     /* [N] */
  float *d_phi_dot; /* [N] */
  float *d_atp;     /* [N] */

  /* Active Inference */
  float *d_phi_target; /* [N] expected phase */
  float *d_error;      /* [N] prediction error */

  /* Sensory: host writes d_sensory_input, kernel copies to d_sensory_forcing */
  float *d_sensory_input;   /* [N] — host-writeable each tick */
  float *d_sensory_forcing; /* [N] — device internal */

  /* CSR topology */
  int *d_csr_row_ptrs;    /* [N+1] */
  int *d_csr_col_indices; /* [E]   */
  float *d_csr_weights;   /* [E]   */
  int *d_delay_ticks;     /* [E]   */

  /* Ring buffers — source-major layout: [src * MAX_DELAY_TICKS + slot] */
  float *d_history_phi;     /* [N * MAX_DELAY_TICKS] */
  float *d_history_phi_dot; /* [N * MAX_DELAY_TICKS] */

  /* Scratch */
  float *d_coupling_sum; /* [N] */
  uint32_t *d_rng;       /* [N] per-thread XORSHIFT state */
};

/* ── Kernel declarations ────────────────────────────────────────────────── */
__global__ void kernel_sensory_endocrine(LeviathanDeviceState state);
__global__ void kernel_synaptic_integration(LeviathanDeviceState state);
__global__ void kernel_kinematics(LeviathanDeviceState state);
__global__ void kernel_plasticity_ptdp(LeviathanDeviceState state);
__global__ void kernel_metabolism(LeviathanDeviceState state);

#endif /* KERNELS_H */
