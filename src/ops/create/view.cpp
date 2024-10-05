#include "ops/create/view.hpp"

#include "core/DataType.hpp"
#include "core/SizeArray.cuh"

namespace ops {

Tensor view(Tensor tensor, Slice::SliceArray slice) {
  SizeArray dims;
  SizeArray strides;
  size_t offset = 0;
  for (size_t i = 0; i < slice.size; i++) {
    if (!slice[i].has_value()) {
      dims.push_back(tensor.getDims()[i]);
      strides.push_back(tensor.getStrides()[i]);
    } else {
      offset += tensor.getStrides()[i] * slice[i].value();
    }
  }
  std::cout << "offset: " << offset << std::endl;
  DeviceArray *data_view =
      new DeviceArray(*tensor.getData() + offset * tensor.getDtype().size);
  std::shared_ptr<DeviceArray> data(tensor.getData(), data_view);

  return Tensor(dims, strides, data, tensor.getDtype());
}

} // namespace ops