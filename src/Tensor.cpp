#include "Tensor.hpp"

Tensor::Tensor(std::vector<size_t> dims, void *data, DataType dtype)
    : dims(dims), data(data), dtype(dtype) {}

Tensor::~Tensor() {}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  os << "Tensor(";
  os << "dims=[";
  for (size_t i = 0; i < tensor.dims.size(); i++) {
    os << tensor.dims[i];
    if (i < tensor.dims.size() - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "data=" << tensor.data << ", ";
  os << "dtype=" << tensor.dtype;
  os << ")";
  return os;
}
