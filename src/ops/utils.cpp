#include "ops/utils.cuh"

/**
 * @return The launch configuration for a 1D kernel.
 */
std::tuple<dim3, dim3> launch_config_1d(int problem_size) {
  dim3 block_dim(256, 1, 1);
  dim3 grid_dim((problem_size + 255) / 256, 1, 1);
  return std::make_tuple(grid_dim, block_dim);
}
