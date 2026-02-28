cmake_minimum_required(VERSION 3.18)
project(Leviathan CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Flags for performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

add_executable(leviathan 
    src/cuda/main.cu
    src/cuda/kernels.cu
)

# Optional: Add testing or other dependencies here if needed
