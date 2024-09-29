#pragma once

#include <cstddef>
#include <iostream>

#ifndef SWITCH_DATATYPE
#define SWITCH_DATATYPE(dtype, template_lambda)                                \
  {                                                                            \
    switch (dtype.tag) {                                                       \
    case DataType::_INT32: {                                                   \
      using T = int32_t;                                                       \
      return template_lambda();                                                \
    }                                                                          \
    case DataType::_FLOAT32: {                                                 \
      using T = float;                                                         \
      return template_lambda();                                                \
    }                                                                          \
    }                                                                          \
  }
#endif

class DataType {
public:
  static const DataType INT32;
  static const DataType FLOAT32;

  enum Enum { _INT32, _FLOAT32 };
  Enum tag;

  const std::string name;
  const size_t size;
  bool operator==(const DataType &dtype) const;
  bool operator!=(const DataType &dtype) const;
  friend std::ostream &operator<<(std::ostream &os, const DataType &dtype);

private:
  DataType(std::string name, size_t size, Enum tag);
  DataType(DataType const &) = delete;
};
