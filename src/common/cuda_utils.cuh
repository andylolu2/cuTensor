#pragma once

#include <tuple>

/**
 * @return The global thread index of the current thread.
 */
__device__ size_t global_thread_index();

/**
 * @return The launch configuration for a 1D kernel.
 */
std::tuple<dim3, dim3> launch_config_1d(int problem_size);