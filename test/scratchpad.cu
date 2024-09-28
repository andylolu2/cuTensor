#include "DataType.hpp"
#include "Tensor.hpp"
#include "detail/Index.cuh"
#include "detail/RuntimeArray.cuh"
#include "ops/add/kernel.cuh"

#include <iostream>

int main(int argc, char const *argv[]) {

  void *a_ptr;
  void *b_ptr;
  void *c_ptr;
  cudaMalloc(&a_ptr, 10 * sizeof(float));
  cudaMalloc(&b_ptr, 10 * sizeof(float));
  cudaMalloc(&c_ptr, 10 * sizeof(float));

  RuntimeArray a({10}, {1}, a_ptr, DataType::FLOAT32);
  RuntimeArray b({10}, {1}, b_ptr, DataType::FLOAT32);
  RuntimeArray c({10}, {1}, c_ptr, DataType::FLOAT32);

  return 0;
}
