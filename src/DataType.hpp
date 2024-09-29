#pragma once

#include <cstddef>
#include <iostream>

#ifndef SWITCH_DATATYPE
#define SWITCH_DATATYPE(dtype, template_lambda)                                \
  {                                                                            \
    if (dtype == DataType::INT32) {                                            \
      using T = int32_t;                                                       \
      return template_lambda();                                                \
    } else if (dtype == DataType::FLOAT32) {                                   \
      using T = float;                                                         \
      return template_lambda();                                                \
    }                                                                          \
    __builtin_unreachable();                                                   \
  }
#endif

class DataType {
public:
  static const DataType INT32;
  static const DataType FLOAT32;

  const std::string name;
  const size_t size;
  bool operator==(const DataType &dtype) const;
  bool operator!=(const DataType &dtype) const;
  friend std::ostream &operator<<(std::ostream &os, const DataType &dtype);

private:
  DataType(std::string name, size_t size);
  DataType(DataType const &) = delete;
};
