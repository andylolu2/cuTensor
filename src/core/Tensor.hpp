#pragma once

#include "core/DataType.hpp"
#include "core/DeviceArray.hpp"
#include "core/SizeArray.cuh"

#include <inttypes.h>
#include <iostream>
#include <memory>
#include <vector>

class Tensor {
private:
  const SizeArray dims;
  const SizeArray strides;
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
      SizeArray dims, SizeArray strides, std::shared_ptr<DeviceArray> data,
      const DataType &dtype
  );
  ~Tensor();

  std::shared_ptr<DeviceArray> getData() const;
  void *getRawData() const;
  size_t getSize() const;
  size_t getNDims() const;
  const DataType &getDtype() const;
  SizeArray getDims() const;
  SizeArray getStrides() const;

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);
};
