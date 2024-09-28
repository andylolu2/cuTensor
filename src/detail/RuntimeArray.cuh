#pragma once

#include "DataType.hpp"
#include "detail/Index.cuh"

#include <cstddef>
#include <vector>

class RuntimeArray {
private:
  void *data;
  DataType _dtype;
  std::vector<size_t> dims;
  std::vector<size_t> strides;

public:
  __host__ __device__ RuntimeArray(
      std::vector<size_t> dims, std::vector<size_t> strides, void *data,
      DataType dtype
  );
  __host__ __device__ ~RuntimeArray();

  __host__ __device__ void *operator()(Index index) const;
  __host__ __device__ std::vector<size_t> shape() const;
  __host__ __device__ size_t size() const;
  __host__ __device__ size_t n_dims() const;
  __host__ __device__ DataType dtype() const;
  friend std::ostream &operator<<(std::ostream &os, const RuntimeArray &tensor);
};