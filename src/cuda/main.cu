/*
 * main.cu — Phase 9: Topology Import + Proper Device Lifecycle
 *
 * Upgrades:
 *   - Removes synthetic ring topology — reads CSR from exported .bin file
 *   - Adds d_rng array for per-thread XORSHIFT noise
 *   - Adds d_sensory_input array for host-writeable sensory forcing
 *   - Single cudaDeviceSynchronize() per tick (not per kernel)
 *   - Full cudaFree() on shutdown
 *   - Streams: stream_kernels (compute) for future overlap
 */
#include "kernels.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

/* ── Topology loader — reads CSR written by Python topology.py ───────────
 * Binary format (little-endian):
 *   [int32] N, [int32] E
 *   [int32 x (N+1)] row_ptr
 *   [int32 x E]     col_idx
 *   [float32 x E]   weights
 *   [int32 x E]     delay_ticks
 *   [float32 x N]   base_omegas
 * ───────────────────────────────────────────────────────────────────────── */
struct HostTopology {
  int N, E;
  std::vector<int> row_ptr, col_idx, delay_ticks;
  std::vector<float> weights, base_omegas;
};

bool load_topology(const char *path, HostTopology &topo) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Cannot open topology: " << path << "\n";
    return false;
  }

  f.read((char *)&topo.N, 4);
  f.read((char *)&topo.E, 4);

  topo.row_ptr.resize(topo.N + 1);
  topo.col_idx.resize(topo.E);
  topo.weights.resize(topo.E);
  topo.delay_ticks.resize(topo.E);
  topo.base_omegas.resize(topo.N);

  f.read((char *)topo.row_ptr.data(), (topo.N + 1) * 4);
  f.read((char *)topo.col_idx.data(), topo.E * 4);
  f.read((char *)topo.weights.data(), topo.E * 4);
  f.read((char *)topo.delay_ticks.data(), topo.E * 4);
  f.read((char *)topo.base_omegas.data(), topo.N * 4);
  return true;
}

/* ── CUDA helper ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = call;                                                      \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA error " << cudaGetErrorString(e) << " at "            \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

template <class T> static void alloc_and_copy(T **d, const std::vector<T> &h) {
  CUDA_CHECK(cudaMalloc(d, h.size() * sizeof(T)));
  CUDA_CHECK(
      cudaMemcpy(*d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

/* ── Main ─────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
  const char *topo_path = (argc > 1) ? argv[1] : "leviathan_topology.bin";
  int num_ticks = (argc > 2) ? atoi(argv[2]) : 1000;

  std::cout << "Leviathan v2.0 CUDA — Phase 9\n";
  std::cout << "Loading topology from: " << topo_path << "\n";

  HostTopology topo;
  if (!load_topology(topo_path, topo))
    return 1;

  int N = topo.N, E = topo.E;
  std::cout << "N=" << N << "  E=" << E << "  ticks=" << num_ticks << "\n";

  /* ── Initialise device state ───────────────────────────────────────── */
  LeviathanDeviceState state = {};
  state.num_nodes = N;
  state.num_edges = E;
  state.current_tick = 0;

  /* Scalars */
  state.DA = 1.0f;
  state.NA = 0.1f;
  state.SER = 1.0f;
  state.ACH = 1.0f;

  /* Topology (imported) */
  alloc_and_copy(&state.d_csr_row_ptrs, topo.row_ptr);
  alloc_and_copy(&state.d_csr_col_indices, topo.col_idx);
  alloc_and_copy(&state.d_csr_weights, topo.weights);
  alloc_and_copy(&state.d_delay_ticks, topo.delay_ticks);
  alloc_and_copy(&state.d_base_omegas, topo.base_omegas);

  /* Node state */
  std::vector<float> zeros_n(N, 0.0f), atp_n(N, 1000.0f);
  alloc_and_copy(&state.d_phi, zeros_n);
  alloc_and_copy(&state.d_phi_dot, zeros_n);
  alloc_and_copy(&state.d_atp, atp_n);

  /* Sensory forcing (host writes here each tick) */
  CUDA_CHECK(cudaMalloc(&state.d_sensory_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state.d_sensory_forcing, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(state.d_sensory_input, 0, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(state.d_sensory_forcing, 0, N * sizeof(float)));

  /* Coupling scratch */
  CUDA_CHECK(cudaMalloc(&state.d_coupling_sum, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(state.d_coupling_sum, 0, N * sizeof(float)));

  /* Ring buffers — source-major: [src * MAX_DELAY_TICKS + slot] */
  CUDA_CHECK(cudaMalloc(&state.d_history_phi,
                        (size_t)N * MAX_DELAY_TICKS * sizeof(float)));
  CUDA_CHECK(cudaMemset(state.d_history_phi, 0,
                        (size_t)N * MAX_DELAY_TICKS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state.d_history_phi_dot,
                        (size_t)N * MAX_DELAY_TICKS * sizeof(float)));
  CUDA_CHECK(cudaMemset(state.d_history_phi_dot, 0,
                        (size_t)N * MAX_DELAY_TICKS * sizeof(float)));

  /* Per-thread RNG seeds */
  std::vector<uint32_t> rng_seeds(N);
  for (int i = 0; i < N; i++)
    rng_seeds[i] = (uint32_t)(i * 2654435761u + 1);
  CUDA_CHECK(cudaMalloc(&state.d_rng, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(state.d_rng, rng_seeds.data(), N * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  /* ── Kernel configuration ─────────────────────────────────────────── */
  int TPB = 256;
  int BPG = (N + TPB - 1) / TPB;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  /* ── Simulation loop ─────────────────────────────────────────────── */
  std::cout << "Running " << num_ticks << " ticks...\n";

  for (int tick = 0; tick < num_ticks; tick++) {
    /* 1. Sensory + Endocrine */
    kernel_sensory_endocrine<<<BPG, TPB, 0, stream>>>(state);

    /* 2. Synaptic integration */
    kernel_synaptic_integration<<<BPG, TPB, 0, stream>>>(state);

    /* 3. Kinematics (RK2) */
    kernel_kinematics<<<BPG, TPB, 0, stream>>>(state);

    /* 4. Plasticity + Metabolism */
    kernel_plasticity_ptdp<<<BPG, TPB, 0, stream>>>(state);
    kernel_metabolism<<<BPG, TPB, 0, stream>>>(state);

    /* ONE synchronise per tick — not per kernel */
    CUDA_CHECK(cudaStreamSynchronize(stream));

    state.current_tick++;
  }

  /* ── Read back results ────────────────────────────────────────────── */
  std::vector<float> h_phi(N);
  CUDA_CHECK(cudaMemcpy(h_phi.data(), state.d_phi, N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::cout << "Done. Node 0 final phase: " << h_phi[0] << "\n";
  std::cout << "DA=" << state.DA << " NA=" << state.NA << " SER=" << state.SER
            << " ACH=" << state.ACH << "\n";

  /* ── Full device free ─────────────────────────────────────────────── */
  cudaFree(state.d_csr_row_ptrs);
  cudaFree(state.d_csr_col_indices);
  cudaFree(state.d_csr_weights);
  cudaFree(state.d_delay_ticks);
  cudaFree(state.d_base_omegas);
  cudaFree(state.d_phi);
  cudaFree(state.d_phi_dot);
  cudaFree(state.d_atp);
  cudaFree(state.d_sensory_input);
  cudaFree(state.d_sensory_forcing);
  cudaFree(state.d_coupling_sum);
  cudaFree(state.d_history_phi);
  cudaFree(state.d_history_phi_dot);
  cudaFree(state.d_rng);
  cudaStreamDestroy(stream);

  return 0;
}
