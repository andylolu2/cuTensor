#pragma once

#include "core/ConstArray.cuh"
#include "core/DataType.hpp"
#include "core/DeviceArray.hpp"

#include <inttypes.h>
#include <iostream>
#include <memory>
#include <vector>

class Tensor {
private:
  const ConstArray dims;
  const ConstArray strides;
  const size_t n_dims;
  const size_t size;
  const DataType &dtype;
  std::shared_ptr<DeviceArray> data;

public:
  Tensor(
      std::initializer_list<size_t> dims, std::initializer_list<size_t> strides,
      std::shared_ptr<DeviceArray> data, const DataType &dtype
  );
  Tensor(
      ConstArray dims, ConstArray strides, std::shared_ptr<DeviceArray> data,
      const DataType &dtype
  );
  ~Tensor();

  void *getData() const;
  size_t getSize() const;
  size_t getNDims() const;
  const DataType &getDtype() const;
  ConstArray getDims() const;
  ConstArray getStrides() const;

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
};

Tensor empty(ConstArray dims, const DataType &dtype);
