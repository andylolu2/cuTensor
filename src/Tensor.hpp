#pragma once

#include "DataType.hpp"
#include "detail/RuntimeArray.cuh"

#include <inttypes.h>
#include <iostream>
#include <vector>

class Tensor {
private:
  std::vector<size_t> dims;
  void *data;
  DataType dtype;

public:
  Tensor(std::vector<size_t> dims, void *data, DataType dtype);
  ~Tensor();

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
};