#include "core/DeviceArray.hpp"

#include "macros.hpp"

#include <cuda_runtime.h>

DeviceArray::DeviceArray(size_t size) : size(size) {
  CUDA_CHECK(cudaMalloc(&data, size));
}

DeviceArray::DeviceArray(void *data, size_t size) : data(data), size(size) {}

DeviceArray::~DeviceArray() { CUDA_CHECK(cudaFree(data)); }

DeviceArray DeviceArray::operator+(size_t offset) const {
  return DeviceArray(static_cast<char *>(data) + offset, size - offset);
}

void *DeviceArray::getData() const { return data; }

size_t DeviceArray::getSize() const { return size; }