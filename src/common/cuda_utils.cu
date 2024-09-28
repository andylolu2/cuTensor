#include "common/cuda_utils.cuh"

__device__ size_t global_thread_index() {
  size_t block_index =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  size_t thread_index_in_block = threadIdx.x + threadIdx.y * blockDim.x +
                                 threadIdx.z * blockDim.x * blockDim.y;
  return block_index * (blockDim.x * blockDim.y * blockDim.z) +
         thread_index_in_block;
}

std::tuple<dim3, dim3> launch_config_1d(int problem_size) {
  dim3 block_dim(256, 1, 1);
  dim3 grid_dim((problem_size + 255) / 256, 1, 1);
  return std::make_tuple(grid_dim, block_dim);
}