#include "core/DeviceArray.hpp"

#include "macros.hpp"

#include <cuda_runtime.h>

DeviceArray::DeviceArray(size_t size) : size(size) {
  CUDA_CHECK(cudaMalloc(&data, size));
}

DeviceArray::~DeviceArray() { CUDA_CHECK(cudaFree(data)); }

void *DeviceArray::getData() const { return data; }

size_t DeviceArray::getSize() const { return size; }