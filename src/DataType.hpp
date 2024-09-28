#pragma once

#include <cstddef>
#include <iostream>

// template_lambda = [&] { some_code_at_uses_t<T>(); }

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
  }
// switch (dtype) {                                                           \
    // case DataType::Type::INT32:                                                \
    //   using T = int32_t;                                                       \
    //   return template_lambda();                                                \
    // case DataType::Type::FLOAT32:                                              \
    //   using T = float;                                                         \
    //   return template_lambda();                                                \
    // }                                                                          \
  }
#endif

enum class DataType { INT32, FLOAT32 };

size_t dtype_size(DataType dtype);

std::string dtype_name(DataType dtype);

std::ostream &operator<<(std::ostream &os, DataType dtype);

// class DataType {
// public:
//   enum class Type { INT32, FLOAT32 };

// private:
//   Type type;

// public:
//   DataType(Type type);
//   ~DataType();

//   size_t size() const;

//   bool operator==(const DataType &dtype) const;
//   bool operator!=(const DataType &dtype) const { return !(*this == dtype); }

//   friend std::ostream &operator<<(std::ostream &os, const DataType &dtype);
// };