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

enum class DataType { INT32, FLOAT32 };

size_t dtype_size(DataType dtype);

std::string dtype_name(DataType dtype);

std::ostream &operator<<(std::ostream &os, DataType dtype);
