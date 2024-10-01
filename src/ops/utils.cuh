#pragma once

#include "common/cuda_utils.hpp"

#include <tuple>

/**
 * @return The launch configuration for a 1D kernel.
 */
std::tuple<dim3, dim3> launch_config_1d(int problem_size);

/**
 * @return The global thread index of the current thread.
 */
DEVICE size_t global_thread_index() {
  size_t block_index =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  size_t thread_index_in_block = threadIdx.x + threadIdx.y * blockDim.x +
                                 threadIdx.z * blockDim.x * blockDim.y;
  return block_index * (blockDim.x * blockDim.y * blockDim.z) +
         thread_index_in_block;
}