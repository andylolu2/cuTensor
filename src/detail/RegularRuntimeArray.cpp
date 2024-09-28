#pragma once

#include "detail/RegularRuntimeArray.hpp"

RegularRuntimeArray::RegularRuntimeArray(
    std::vector<size_t> dims, std::vector<size_t> strides, void *data,
    DataType dtype
)
    : RuntimeArray(dims, dtype), strides(strides), data(data) {}

RegularRuntimeArray::~RegularRuntimeArray() {}

size_t RegularRuntimeArray::operator()(Index index) const {
  size_t offset = 0;
  for (size_t i = 0; i < index.n_dims(); i++) {
    offset += index[i] * strides[i];
  }
  return offset;
}
