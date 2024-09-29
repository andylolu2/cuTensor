#include "DataType.hpp"
#include "Tensor.hpp"
#include "ops/add/host.hpp"
#include "ops/fill/host.hpp"

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

  fill_tensor(a, 1.24);
  fill_tensor(b, 2.0);

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c (before): " << c << std::endl;

  add_tensors(a, b, c);

  std::cout << "c (after): " << c << std::endl;

  cudaFree(a_ptr);
  cudaFree(b_ptr);
  cudaFree(c_ptr);

  return 0;
}
