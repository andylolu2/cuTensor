#include "DataType.hpp"

// size_t dtype_size(DataType dtype) {
//   SWITCH_DATATYPE(dtype, [&] { return sizeof(T); });
// }

// std::string dtype_name(DataType dtype) {
//   SWITCH_DATATYPE(dtype, [&] {
//     std::string name = "";
//     if (dtype == DataType::INT32) {
//       name = "int32";
//     } else if (dtype == DataType::FLOAT32) {
//       name = "fp32";
//     }
//     return name;
//   });
// }

const DataType DataType::INT32 = DataType("int32", sizeof(int32_t));
const DataType DataType::FLOAT32 = DataType("fp32", sizeof(float));

DataType::DataType(std::string name, size_t size) : name(name), size(size) {}

bool DataType::operator==(const DataType &dtype) const {
  return &dtype == this;
}

bool DataType::operator!=(const DataType &dtype) const {
  return &dtype != this;
}

std::ostream &operator<<(std::ostream &os, const DataType &dtype) {
  os << dtype.name;
  return os;
}
