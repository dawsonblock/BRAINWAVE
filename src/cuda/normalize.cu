/*
 * normalize.cu — Phase 10: GPU homeostatic row-normalization
 *
 * Two-pass approach (matching Python's per-row normalization):
 *   Pass 1: row_sum_kernel accumulates incoming weight sums per row
 *   Pass 2: normalize_kernel scales weights where row_sum > W_MAX
 */
#include <cuda_runtime.h>

__global__ void
row_sum_kernel(int E, const int *__restrict__ dst_idx,
               const float *__restrict__ weights,
               float *__restrict__ row_sums /* [N], pre-zeroed */
) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E)
    return;
  atomicAdd(&row_sums[dst_idx[e]], weights[e]);
}

__global__ void normalize_weights(int E, const int *__restrict__ dst_idx,
                                  float *__restrict__ weights,
                                  const float *__restrict__ row_sums,
                                  float max_sum) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E)
    return;

  int dst = dst_idx[e];
  float s = row_sums[dst];

  if (s > max_sum) {
    weights[e] *= max_sum / s;
  }
}

/* Host wrapper — keeps main.cu clean */
void gpu_row_normalize(int E, const int *dst_idx, float *weights,
                       float *d_row_sums_scratch, /* [N] — caller allocates */
                       int N, float max_sum, cudaStream_t stream) {
  int TPB = 256;
  int BPG = (E + TPB - 1) / TPB;

  cudaMemsetAsync(d_row_sums_scratch, 0, N * sizeof(float), stream);
  row_sum_kernel<<<BPG, TPB, 0, stream>>>(E, dst_idx, weights,
                                          d_row_sums_scratch);
  normalize_weights<<<BPG, TPB, 0, stream>>>(E, dst_idx, weights,
                                             d_row_sums_scratch, max_sum);
}
