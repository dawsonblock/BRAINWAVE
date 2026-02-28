/*
 * fused_step.cu — Phase 10: Fused edge-accumulate + RK2 integrate kernel
 *
 * Fuses the synaptic integration (coupling accumulation) and kinematics
 * update into a single kernel pass, reducing global memory traffic by
 * eliminating one write + one read of d_coupling_sum.
 *
 * Strategy:
 *   - Shared memory holds per-block coupling partial sums
 *   - Edge threads accumulate into smem with atomicAdd
 *   - After sync, node thread i reads its coupling and runs RK2
 *
 * Launch: <<<ceil(max(N,E)/TPB), TPB, TPB*sizeof(float)>>>
 *
 * Limitation: block must fit N nodes in shared memory (works for N ≤ 8192
 * with float32 = 32KB). For larger N, use the separate kernel path.
 */
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void fused_step_kernel(
    int N, int E, float *__restrict__ phi, float *__restrict__ phi_dot,
    const float *__restrict__ history, /* [N * MAX_DELAY] source-major */
    int history_index, int max_delay, const int *__restrict__ dst_idx, /* [E] */
    const int *__restrict__ col_idx,   /* [E] source per edge */
    const int *__restrict__ delay,     /* [E] */
    const float *__restrict__ weights, /* [E] */
    float dt, float damping, float inertia, float velocity_clamp,
    float ser /* serotonin coupling scale */
) {
  extern __shared__ float smem[]; /* [blockDim.x] partial coupling */

  int tid = threadIdx.x;
  int stride = blockDim.x;

  /* Zero shared coupling buffer */
  smem[tid] = 0.0f;
  __syncthreads();

  /* ── Phase 1: Edge accumulation ───────────────────────────────────── */
  for (int e = blockIdx.x * stride + tid; e < E; e += gridDim.x * stride) {
    int dst = dst_idx[e];
    int src = col_idx[e];
    int slot = ((history_index - delay[e]) % max_delay + max_delay) % max_delay;

    float phi_src = history[src * max_delay + slot];
    float phi_dst = phi[dst];

    float contrib = weights[e] * __sinf(phi_src - phi_dst);
    atomicAdd(&smem[dst % stride], contrib); /* Local smem bucket */
  }
  __syncthreads();

  /* ── Phase 2: RK2 integration (one thread per node) ──────────────── */
  for (int i = blockIdx.x * stride + tid; i < N; i += gridDim.x * stride) {
    float coup = smem[i % stride] * ser;

    float pd = phi_dot[i];
    float p = phi[i];

    float a1 = (coup - damping * pd) / inertia;
    float pd_mid = pd + 0.5f * dt * a1;
    float a2 = (coup - damping * pd_mid) / inertia;

    float pd_new = pd + dt * a2;
    pd_new = fminf(fmaxf(pd_new, -velocity_clamp), velocity_clamp);

    float p_new = p + dt * pd_mid;
    p_new = fmodf(p_new, 2.0f * CUDART_PI_F);
    if (p_new < 0.0f)
      p_new += 2.0f * CUDART_PI_F;

    phi[i] = p_new;
    phi_dot[i] = pd_new;
  }
}
