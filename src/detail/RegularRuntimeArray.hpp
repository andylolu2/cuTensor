#pragma once

#include "detail/RuntimeArray.cuh"

class RegularRuntimeArray : RuntimeArray {
private:
  void *data;
  std::vector<size_t> strides;

public:
  RegularRuntimeArray(
      std::vector<size_t> dims, std::vector<size_t> strides, void *data,
      DataType dtype
  );
  ~RegularRuntimeArray();

  size_t operator()(Index index) const;
};