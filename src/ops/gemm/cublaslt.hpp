#pragma once

#include "core/Tensor.hpp"

#include <cublasLt.h>

namespace ops {

namespace detail {

template <typename T>
struct CublasLtGemm {
  cublasLtHandle_t lightHandle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
  cudaStream_t stream;

  CublasLtGemm(Tensor output, Tensor a, Tensor b);

  ~CublasLtGemm();

  void set_matrix_order(cublasLtMatrixLayout_t &desc, Tensor tensor);
  void check_args(Tensor output, Tensor a, Tensor b);
};
} // namespace detail

void gemm(Tensor output, Tensor a, Tensor b);

} // namespace ops