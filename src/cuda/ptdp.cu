/*
 * ptdp.cu — Phase 10: GPU PTDP plasticity kernel
 *
 * Fully edge-parallel PTDP with:
 *   - atan2 phase wrapping (identical to Python)
 *   - DA-gated LTP magnitude
 *   - Weight non-negativity clamp
 * Followed by row_normalize (see normalize.cu) for homeostasis.
 */
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void ptdp_update(
    int E, const int *__restrict__ dst_idx, const int *__restrict__ col_idx,
    const float *__restrict__ phi,         /* current phase (N,) */
    const float *__restrict__ phi_delayed, /* delayed source phase (E,) */
    float *__restrict__ weights, float dopamine, float lr_plus, float lr_minus,
    float tau_plus, float tau_minus, float dt) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E)
    return;

  float phi_dst = phi[dst_idx[e]];
  float delta = phi_delayed[e] - phi_dst;

  /* atan2 wrapping → [-π, π] */
  delta = atan2f(__sinf(delta), __cosf(delta));

  float dw;
  if (delta > 0.0f) {
    /* Src leads dst — LTP */
    dw = dopamine * lr_plus * expf(-delta / tau_plus);
  } else {
    /* Dst leads src — LTD */
    dw = -lr_minus * expf(delta / tau_minus);
  }

  /* Dale's Principle Sign Enforcement */
  float old_w = weights[e];
  float new_w = old_w + dw * dt;

  if (old_w > 0.0f) {
    /* Excitatory: stay positive */
    if (new_w < 0.0f)
      new_w = 0.0f;
  } else if (old_w < 0.0f) {
    /* Inhibitory: stay negative */
    if (new_w > 0.0f)
      new_w = 0.0f;
  }

  weights[e] = new_w;
}
