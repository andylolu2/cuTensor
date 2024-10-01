#include "core/DataType.hpp"

const DataType DataType::INT32 =
    DataType("int32", sizeof(int32_t), DataType::_INT32);
const DataType DataType::FLOAT32 =
    DataType("fp32", sizeof(float), DataType::_FLOAT32);

DataType::DataType(std::string name, size_t size, Enum tag)
    : name(name), size(size), tag(tag) {}

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
