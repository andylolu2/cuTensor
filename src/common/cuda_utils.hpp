#pragma once

#include <cstddef>
#include <tuple>

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define HOST_DEVICE __device__ __host__
#define DEVICE __device__ __forceinline__
#elif defined(__CUDACC_RTC__)
#define HOST_DEVICE __device__
#define DEVICE __device__
#else
#define HOST_DEVICE inline
#define DEVICE inline
#endif

/**
 * @return The launch configuration for a 1D kernel.
 */
std::tuple<dim3, dim3> launch_config_1d(int problem_size);