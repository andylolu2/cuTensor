#pragma once

#include <cstddef>
#include <vector>

class Index {
private:
  using iterator = std::vector<size_t>::const_iterator;
  std::vector<size_t> coord;

public:
  __host__ __device__ Index(std::vector<size_t> coord);
  __host__ __device__ ~Index();

  __host__ __device__ static Index
  from_1d(size_t index_1d, std::vector<size_t> dims) {
    std::vector<size_t> coord(dims.size());
    size_t index_1d_ = index_1d;
    for (size_t i = dims.size() - 1; i > 0; i--) {
      coord[i] = index_1d_ % dims[i];
      index_1d_ /= dims[i];
    }
    return Index(coord);
  };

  __host__ __device__ size_t n_dims() const;
  __host__ __device__ size_t operator[](size_t i) const;
};
