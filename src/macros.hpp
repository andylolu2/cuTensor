#pragma once

#include <cublas.h>
#include <iostream>

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

#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "Got cuda errror: " << cudaGetErrorString(error)            \
                << " at: " << __FILE__ << ":" << __LINE__ << " in "            \
                << __func__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUBLAS_CHECK(status)                                                   \
  {                                                                            \
    cublasStatus_t error = status;                                             \
    if (error != CUBLAS_STATUS_SUCCESS) {                                      \
      std::cerr << "Got cublas errror: " << error << " at: " << __FILE__       \
                << ":" << __LINE__ << " in " << __func__ << std::endl;         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
