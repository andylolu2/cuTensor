#include "core/Tensor.hpp"

Tensor::Tensor(
    std::initializer_list<size_t> dims, std::initializer_list<size_t> strides,
    std::shared_ptr<DeviceArray> data, const DataType &dtype
)
    : n_dims(dims.size()), dtype(dtype), data(std::move(data)), dims(dims),
      strides(strides), size(prod(dims)) {
  if (dims.size() != strides.size()) {
    throw std::invalid_argument("Number of dimensions and strides must match");
  }
}

Tensor::Tensor(
    ConstArray dims, ConstArray strides, std::shared_ptr<DeviceArray> data,
    const DataType &dtype
)
    : n_dims(dims.size), dtype(dtype), data(std::move(data)), dims(dims),
      strides(strides), size(prod(dims)) {
  if (dims.size != strides.size) {
    throw std::invalid_argument("Number of dimensions and strides must match");
  }
}

Tensor::~Tensor() {}

void *Tensor::getData() const { return data->getData(); }
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
                          host.data(), tensor.getData(),
                          tensor.size * sizeof(T), cudaMemcpyDeviceToHost
                      );
                      for (size_t i = 0; i < tensor.size; i++) {
                        os << host[i];
                        if (i < tensor.size - 1) {
                          os << ", ";
                        }
                      }
                      os << "]";
                    }));
  } else if (tensor.n_dims == 2) {
    SWITCH_DATATYPE(tensor.dtype, ([&] {
                      os << ", data=[";
                      //   copy data to host
                      std::vector<T> host(tensor.size);
                      cudaMemcpy(
                          host.data(), tensor.getData(),
                          tensor.size * sizeof(T), cudaMemcpyDeviceToHost
                      );
                      os << std::endl;
                      for (size_t i = 0; i < tensor.dims[0]; i++) {
                        os << "[";
                        for (size_t j = 0; j < tensor.dims[1]; j++) {
                          os << host[i * tensor.dims[1] + j];
                          if (j < tensor.dims[1] - 1) {
                            os << ", ";
                          }
                        }
                        os << "]";
                        if (i < tensor.dims[0] - 1) {
                          os << ", " << std::endl;
                        }
                      }
                      os << "]";
                    }));
  }
  os << ")";
  return os;
}

Tensor empty(ConstArray dims, const DataType &dtype) {
  size_t size = prod(dims);
  ConstArray strides(dims.size, 1);
  for (size_t i = dims.size - 1; i-- > 0;) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  std::shared_ptr<DeviceArray> data(
      std::make_shared<DeviceArray>(size * dtype.size)
  );
  return Tensor(dims, strides, data, dtype);
}
