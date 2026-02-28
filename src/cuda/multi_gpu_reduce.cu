/*
 * multi_gpu_reduce.cu — Phase 10: NCCL AllReduce for multi-GPU coupling
 *
 * After each GPU computes its coupling partial sum:
 *   nccl_allreduce_coupling() sums contributions across all GPUs.
 *
 * Each process must call ncclCommInitRank() before the first sim tick.
 * Stream overlap: ensure compute_stream and comm_stream are separate.
 */
#include <cuda_runtime.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_NCCL
void nccl_allreduce_coupling(
    float *d_coupling, /* [N] on this GPU — in-place sum */
    int N, ncclComm_t comm, cudaStream_t stream) {
  ncclAllReduce((const void *)d_coupling, (void *)d_coupling, N, ncclFloat,
                ncclSum, comm, stream);
}

/* Convenience wrapper: initialise NCCL communicator for `world_size` GPUs.
   Call once at startup on each process with rank = local GPU index. */
ncclComm_t nccl_init(int rank, int world_size) {
  ncclComm_t comm;
  ncclUniqueId id;
  if (rank == 0)
    ncclGetUniqueId(&id);
  /* Distribute id to other ranks via MPI or shared file (not shown) */
  ncclCommInitRank(&comm, world_size, id, rank);
  return comm;
}

void nccl_destroy(ncclComm_t comm) { ncclCommDestroy(comm); }

#else

/* Stub for single-GPU builds (no NCCL) */
void nccl_allreduce_coupling(float *, int, void *, void *) {}

#endif /* USE_NCCL */
