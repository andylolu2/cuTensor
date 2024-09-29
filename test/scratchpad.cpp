#include "DataType.hpp"
#include "Tensor.hpp"
#include "ops/add/host.hpp"
#include "ops/fill/host.hpp"

#include <cuda_runtime.h>
#include <iostream>

void print_device(void *ptr, size_t size) {
  std::vector<float> host(size);
  cudaMemcpy(host.data(), ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; i++) {
    std::cout << host[i] << " ";
  }
  std::cout << std::endl;
}

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

  fill_tensor(a, 1.0);
  fill_tensor(b, 2.0);
  print_device(a_ptr, 10);
  print_device(b_ptr, 10);

  std::cout << a << std::endl << b << std::endl << c << std::endl;

  add_tensors(a, b, c);
  print_device(c_ptr, 10);

  cudaFree(a_ptr);
  cudaFree(b_ptr);
  cudaFree(c_ptr);

  return 0;
}
