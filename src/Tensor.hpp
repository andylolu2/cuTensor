#pragma once

#include "ConstArray.cuh"
#include "DataType.hpp"

#include <inttypes.h>
#include <iostream>
#include <vector>

class Tensor {
private:
  ConstArray dims;
  ConstArray strides;
  size_t n_dims;
  size_t size;
  DataType dtype;
  void *data;

public:
  Tensor(
      std::initializer_list<size_t> dims, std::initializer_list<size_t> strides,
      void *data, DataType dtype
  );
  ~Tensor();

  void *getData() const;
  size_t getSize() const;
  size_t getNDims() const;
  DataType getDtype() const;
  ConstArray getDims() const;
  ConstArray getStrides() const;

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
};