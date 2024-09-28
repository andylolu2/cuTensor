#include "DataType.hpp"

// DataType::DataType(Type type) : type(type) {}

// DataType::~DataType() {}

// size_t DataType::size() const {
//   SWITCH_DATATYPE(type, [&] { return sizeof(T); });
// }

// bool DataType::operator==(const DataType &dtype) const {
//   return type == dtype.type;
// }
// bool DataType::operator!=(const DataType &dtype) const {
//   return !(*this == dtype);
// }

// std::ostream &operator<<(std::ostream &os, const DataType &dtype) {
//   switch (dtype.type) {
//   case DataType::Type::FLOAT32:
//     os << "FLOAT32";
//     break;
//   }
//   return os;
// }

size_t dtype_size(DataType dtype) {
  SWITCH_DATATYPE(dtype, [&] { return sizeof(T); });
}

std::string dtype_name(DataType dtype) {
  SWITCH_DATATYPE(dtype, [&] {
    std::string name = "";
    if (dtype == DataType::INT32) {
      name = "INT32";
    } else if (dtype == DataType::FLOAT32) {
      name = "FLOAT32";
    }
    return name;
  });
}

std::ostream &operator<<(std::ostream &os, DataType dtype) {
  os << dtype_name(dtype);
  return os;
}
