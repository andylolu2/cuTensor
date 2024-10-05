#include "core/SizeArray.cuh"

HOST_DEVICE size_t prod(const SizeArray &array) {
  size_t result = 1;
  for (size_t i = 0; i < array.size; i++) {
    result *= array[i];
  }
  return result;
}