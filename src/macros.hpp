#pragma once

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
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
