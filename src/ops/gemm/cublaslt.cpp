#include "ops/gemm/cublaslt.hpp"

#include "macros.hpp"

#include <cublasLt.h>

namespace ops {
namespace detail {

template <>
CublasLtGemm<float>::CublasLtGemm(Tensor output, Tensor a, Tensor b) {
  CUBLAS_CHECK(cublasLtCreate(&lightHandle));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
  );
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Adesc, CUDA_R_32F, a.getDims()[0], a.getDims()[1], a.getStrides()[0]
  ));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Bdesc, CUDA_R_32F, b.getDims()[0], b.getDims()[1], b.getStrides()[0]
  ));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Cdesc, CUDA_R_32F, output.getDims()[0], output.getDims()[1],
      output.getStrides()[0]
  ));
  set_matrix_order(Adesc, a);
  set_matrix_order(Bdesc, b);
  set_matrix_order(Cdesc, output);
};

template <>
CublasLtGemm<half>::CublasLtGemm(Tensor output, Tensor a, Tensor b) {
  CUBLAS_CHECK(cublasLtCreate(&lightHandle));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F)
  );
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Adesc, CUDA_R_16F, a.getDims()[0], a.getDims()[1], a.getStrides()[0]
  ));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Bdesc, CUDA_R_16F, b.getDims()[0], b.getDims()[1], b.getStrides()[0]
  ));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Cdesc, CUDA_R_16F, output.getDims()[0], output.getDims()[1],
      output.getStrides()[0]
  ));
  set_matrix_order(Adesc, a);
  set_matrix_order(Bdesc, b);
  set_matrix_order(Cdesc, output);
};

template <typename T>
CublasLtGemm<T>::CublasLtGemm(Tensor output, Tensor a, Tensor b) {
  throw std::invalid_argument(
      "Unsupported GEMM datatype: " + output.getDtype().name
  );
};

template <typename T>
CublasLtGemm<T>::~CublasLtGemm() {
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
  CUBLAS_CHECK(cublasLtDestroy(lightHandle));
  CUDA_CHECK(cudaStreamDestroy(stream));
};

template <typename T>
void CublasLtGemm<T>::set_matrix_order(
    cublasLtMatrixLayout_t &desc, Tensor tensor
) {
  if (tensor.getNDims() != 2) {
    throw std::invalid_argument("Only 2D tensors are supported");
  }
  if (tensor.getStrides()[0] == 1) {
    cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_order, sizeof(col_order)
    ));
  } else if (tensor.getStrides()[1] == 1) {
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)
    ));
  } else {
    throw std::invalid_argument("Only contiguous tensors are supported");
  }
}

} // namespace detail

void gemm(Tensor output, Tensor a, Tensor b) {
  SWITCH_DATATYPE(output.getDtype(), ([&] {
                    detail::CublasLtGemm<T> gemm(output, a, b);
                    T alpha(1), beta(0);
                    CUBLAS_CHECK(cublasLtMatmul(
                        gemm.lightHandle, gemm.matmulDesc, &alpha, a.getData(),
                        gemm.Adesc, b.getData(), gemm.Bdesc, &beta,
                        output.getData(), gemm.Cdesc, output.getData(),
                        gemm.Cdesc, nullptr, nullptr, 0, gemm.stream
                    ));
                  }));
}

} // namespace ops