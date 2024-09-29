#include "Tensor.hpp"

Tensor::Tensor(
    std::initializer_list<size_t> dims, std::initializer_list<size_t> strides,
    void *data, const DataType &dtype
)
    : n_dims(dims.size()), dtype(dtype), data(data), dims(dims),
      strides(strides), size(prod(dims)) {
  if (dims.size() != strides.size()) {
    throw std::invalid_argument("Number of dimensions and strides must match");
  }
}

Tensor::~Tensor() {}

void *Tensor::getData() const { return data; }
size_t Tensor::getSize() const { return size; }
size_t Tensor::getNDims() const { return n_dims; }
const DataType &Tensor::getDtype() const { return dtype; }
ConstArray Tensor::getDims() const { return dims; }
ConstArray Tensor::getStrides() const { return strides; }

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  os << "Tensor(";
  os << "dims=[";
  for (size_t i = 0; i < tensor.n_dims; i++) {
    os << tensor.dims[i];
    if (i < tensor.n_dims - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "strides=[";
  for (size_t i = 0; i < tensor.n_dims; i++) {
    os << tensor.strides[i];
    if (i < tensor.n_dims - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "dtype=" << tensor.dtype;

  if (tensor.n_dims == 1) {
    SWITCH_DATATYPE(tensor.dtype, ([&] {
                      os << ", data=[";
                      //   copy data to host
                      std::vector<T> host(tensor.size);
                      cudaMemcpy(
                          host.data(), tensor.data, tensor.size * sizeof(T),
                          cudaMemcpyDeviceToHost
                      );
                      for (size_t i = 0; i < tensor.size; i++) {
                        os << host[i];
                        if (i < tensor.size - 1) {
                          os << ", ";
                        }
                      }
                      os << "]";
                    }));
  }
  os << ")";
  return os;
}
