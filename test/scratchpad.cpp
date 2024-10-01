#include "core/DataType.hpp"
#include "core/Tensor.hpp"
#include "ops/elementwise/funcs.hpp"

#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char const *argv[]) {

  void *a_ptr;
  void *b_ptr;
  void *c_ptr;
  cudaMalloc(&a_ptr, 10 * sizeof(float));
  cudaMalloc(&b_ptr, 10 * sizeof(float));
  cudaMalloc(&c_ptr, 10 * sizeof(float));

  Tensor a({10}, {1}, a_ptr, DataType::FLOAT32);
  Tensor b({10}, {1}, b_ptr, DataType::FLOAT32);
  Tensor c({10}, {1}, c_ptr, DataType::FLOAT32);

  ops::fill(a, 1.24);
  //   ops::fill(b, 2.0);
  ops::rand(b, 0);

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c (before): " << c << std::endl;

  ops::add(c, a, b);

  std::cout << "c (after): " << c << std::endl;

  cudaFree(a_ptr);
  cudaFree(b_ptr);
  cudaFree(c_ptr);

  return 0;
}
