#pragma once

#include "detail/Index.hpp"

#include <cstddef>
#include <functional>
#include <vector>

/* Mapping of logical index to physical index
 *
 * Logical indices are in row-major order.
 */
template <typename T>
class LogicalArray {
private:
  std::vector<size_t> dims;

public:
  virtual T operator[](Index index) const = 0;
  virtual LogicalArray<T> flatten() const = 0;

  std::vector<size_t> shape() const { return dims; }

  size_t size() const {
    size_t size = 1;
    for (size_t dim : dims) {
      size *= dim;
    }
    return size;
  }
};

template <typename T>
class LambdaLogicalArray : LogicalArray<T> {
private:
  std::function<size_t(Index)> index_to_offset;
  T *data;

public:
  LambdaLogicalArray(
      std::function<size_t(Index)> index_to_offset, T *data,
      std::vector<size_t> dims
  )
      : index_to_offset(index_to_offset), data(data), dims_(dims) {}

  T operator[](Index index) const {
    for (size_t i = 0; i < index.size(); i++) {
      if (index[i] >= dims_[i]) {
        throw std::out_of_range("Index out of range");
      }
    }
    return data[index_to_offset(index)];
  }

  LogicalArray<T> flatten() const {
    auto flatten_dims = std::vector<size_t>{size()};
    auto flatten_index_to_offset = [this](Index index_1d) {
      return index_to_offset(Index(index_1d, dims_));
    };
    return LambdaLogicalArray(flatten_index_to_offset, data, flatten_dims);
  }
};

template <typename T>
class RegularLogicalArray : LogicalArray<T> {
private:
  T *data;
  std::vector<size_t> strides;

public:
  RegularLogicalArray(
      T *data, std::vector<size_t> dims, std::vector<size_t> strides
  )
      : data(data), dims_(dims), strides(strides) {}

  T operator[](Index index) const {
    size_t offset = 0;
    for (size_t i = 0; i < index.n_dims(); i++) {
      offset += index[i] * strides[i];
    }
    return data[offset];
  }

  LogicalArray<T> flatten() const {
    auto flatten_dims = std::vector<size_t>{size()};
    auto flatten_index_to_offset = [this](Index index_1d) {
      return this[Index(index_1d, dims_)];
    };
    return LambdaLogicalArray(flatten_index_to_offset, data, flatten_dims);
  }
};