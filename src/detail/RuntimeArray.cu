#include "detail/RuntimeArray.cuh"

__host__ __device__ RuntimeArray::RuntimeArray(
    std::vector<size_t> dims, std::vector<size_t> strides, void *data,
    DataType dtype
)
    : dims(dims), strides(strides), data(data), _dtype(dtype) {}

__host__ __device__ RuntimeArray::~RuntimeArray() {}

__host__ __device__ void *RuntimeArray::operator()(Index index) const {
  size_t offset = 0;
  for (size_t i = 0; i < index.n_dims(); i++) {
    offset += index[i] * strides[i];
  }
  return data + offset * dtype_size(dtype());
}

__host__ __device__ std::vector<size_t> RuntimeArray::shape() const {
  return dims;
}

__host__ __device__ size_t RuntimeArray::size() const {
  size_t size = 1;
  for (size_t dim : dims) {
    size *= dim;
  }
  return size;
}

__host__ __device__ size_t RuntimeArray::n_dims() const { return dims.size(); }

__host__ __device__ DataType RuntimeArray::dtype() const { return _dtype; }

std::ostream &operator<<(std::ostream &os, const RuntimeArray &tensor) {
  os << "RuntimeArray(";
  os << "dims=[";
  for (size_t i = 0; i < tensor.shape().size(); i++) {
    os << tensor.shape()[i];
    if (i < tensor.shape().size() - 1) {
      os << ", ";
    }
  }
  os << "], ";
  os << "dtype=" << tensor.dtype();
  os << ")";
  return os;
}
