#include "ops/create/empty.hpp"

namespace ops {

Tensor empty(SizeArray dims, const DataType &dtype) {
  size_t size = prod(dims);
  SizeArray strides(dims.size, 1);
  for (size_t i = dims.size - 1; i-- > 0;) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  std::shared_ptr<DeviceArray> data(
      std::make_shared<DeviceArray>(size * dtype.size)
  );
  return Tensor(dims, strides, data, dtype);
}

}