#include "kernels.h"
#include <iostream>
#include <random>
#include <vector>

void initialize_state(LeviathanDeviceState &state, int nodes, int num_edges) {
  // Memory allocation helper based on the spec
  state.num_nodes = nodes;
  state.num_edges = num_edges;
  state.current_tick = 0;

  // Default endocrines
  state.DA = 1.0f;
  state.NA = 0.1f;
  state.SER = 1.0f;
  state.ACH = 1.0f;

  // Allocate host memory
  float *h_base_omegas = new float[nodes];
  float *h_phi = new float[nodes];
  float *h_phi_dot = new float[nodes];
  float *h_atp = new float[nodes];
  float *h_sensory_forcing = new float[nodes];

  for (int i = 0; i < nodes; ++i) {
    h_base_omegas[i] = 10.0f; // Simplified
    h_phi[i] = 0.0f;
    h_phi_dot[i] = 0.0f;
    h_atp[i] = 1000.0f;
    h_sensory_forcing[i] = 0.0f;
  }

  // Dummy Sparse Graph (Ring topology for testing)
  int *h_csr_row_ptrs = new int[nodes + 1];
  int *h_csr_col_indices = new int[num_edges];
  float *h_csr_weights = new float[num_edges];
  int *h_delay_ticks = new int[num_edges];

  for (int i = 0; i <= nodes; i++) {
    h_csr_row_ptrs[i] = i * 2; // Assuming 2 edges per node
  }

  for (int e = 0; e < num_edges; e++) {
    h_csr_col_indices[e] = (e + 1) % nodes; // connects to next node roughly
    h_csr_weights[e] = 0.5f;
    h_delay_ticks[e] = 10; // 0.1s delay
  }

  // Float allocations
  cudaMalloc(&state.d_base_omegas, nodes * sizeof(float));
  cudaMemcpy(state.d_base_omegas, h_base_omegas, nodes * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_phi, nodes * sizeof(float));
  cudaMemcpy(state.d_phi, h_phi, nodes * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_phi_dot, nodes * sizeof(float));
  cudaMemcpy(state.d_phi_dot, h_phi_dot, nodes * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_atp, nodes * sizeof(float));
  cudaMemcpy(state.d_atp, h_atp, nodes * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_sensory_forcing, nodes * sizeof(float));
  cudaMemcpy(state.d_sensory_forcing, h_sensory_forcing, nodes * sizeof(float),
             cudaMemcpyHostToDevice);

  // Sparse Matrix Allocations
  cudaMalloc(&state.d_csr_row_ptrs, (nodes + 1) * sizeof(int));
  cudaMemcpy(state.d_csr_row_ptrs, h_csr_row_ptrs, (nodes + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_csr_col_indices, num_edges * sizeof(int));
  cudaMemcpy(state.d_csr_col_indices, h_csr_col_indices,
             num_edges * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_csr_weights, num_edges * sizeof(float));
  cudaMemcpy(state.d_csr_weights, h_csr_weights, num_edges * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMalloc(&state.d_delay_ticks, num_edges * sizeof(int));
  cudaMemcpy(state.d_delay_ticks, h_delay_ticks, num_edges * sizeof(int),
             cudaMemcpyHostToDevice);

  // Ring Buffer Allocations
  cudaMalloc(&state.d_history_phi, MAX_DELAY_TICKS * nodes * sizeof(float));
  cudaMemset(state.d_history_phi, 0, MAX_DELAY_TICKS * nodes * sizeof(float));

  cudaMalloc(&state.d_history_phi_dot, MAX_DELAY_TICKS * nodes * sizeof(float));
  cudaMemset(state.d_history_phi_dot, 0,
             MAX_DELAY_TICKS * nodes * sizeof(float));

  cudaMalloc(&state.d_coupling_sum, nodes * sizeof(float));
  cudaMemset(state.d_coupling_sum, 0, nodes * sizeof(float));

  // Free host arrays
  delete[] h_base_omegas;
  delete[] h_phi;
  delete[] h_phi_dot;
  delete[] h_atp;
  delete[] h_sensory_forcing;
  delete[] h_csr_row_ptrs;
  delete[] h_csr_col_indices;
  delete[] h_csr_weights;
  delete[] h_delay_ticks;
}

int main(int argc, char **argv) {
  std::cout << "Starting Leviathan v2.0 CUDA Simulator Engine" << std::endl;

  LeviathanDeviceState state;
  int num_nodes = 1000;
  int num_edges = num_nodes * 2;

  initialize_state(state, num_nodes, num_edges);

  // Launch configuration
  int threadsPerBlock = 256;
  int blocksPerGridNodes = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
  // PTDP kernel might run over nodes or edges, we designed it over nodes

  int num_ticks = 1000; // 10 seconds of simulated time
  std::cout << "Executing " << num_ticks << " kernel ticks over " << num_nodes
            << " nodes..." << std::endl;

  for (int tick = 0; tick < num_ticks; ++tick) {
    // Run Kernels Sequence:

    // 1. Sensory & Endocrine
    kernel_sensory_endocrine<<<blocksPerGridNodes, threadsPerBlock>>>(state);
    cudaDeviceSynchronize();

    // 2. Integration
    kernel_synaptic_integration<<<blocksPerGridNodes, threadsPerBlock>>>(state);
    cudaDeviceSynchronize();

    // 3. Kinematics
    kernel_kinematics<<<blocksPerGridNodes, threadsPerBlock>>>(state);
    cudaDeviceSynchronize();

    // 4. PTDP & Metabolism
    kernel_plasticity_ptdp<<<blocksPerGridNodes, threadsPerBlock>>>(state);
    kernel_metabolism<<<blocksPerGridNodes, threadsPerBlock>>>(state);
    cudaDeviceSynchronize();

    state.current_tick++;
  }

  // Copy final state back to CPU for verification
  float *h_final_phi = new float[num_nodes];
  cudaMemcpy(h_final_phi, state.d_phi, num_nodes * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Simulation loop completed smoothly!" << std::endl;
  std::cout << "Sample Node 0 Final Phase: " << h_final_phi[0] << std::endl;

  delete[] h_final_phi;

  // Normally you would cudaFree everything here

  return 0;
}
