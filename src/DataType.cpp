#include "DataType.hpp"

size_t dtype_size(DataType dtype) {
  SWITCH_DATATYPE(dtype, [&] { return sizeof(T); });
}

std::string dtype_name(DataType dtype) {
  SWITCH_DATATYPE(dtype, [&] {
    std::string name = "";
    if (dtype == DataType::INT32) {
      name = "int32";
    } else if (dtype == DataType::FLOAT32) {
      name = "fp32";
    }
    return name;
  });
}

std::ostream &operator<<(std::ostream &os, DataType dtype) {
  os << dtype_name(dtype);
  return os;
}
