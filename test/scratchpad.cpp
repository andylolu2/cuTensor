#include "core/DataType.hpp"
#include "core/Tensor.hpp"
#include "macros.hpp"
#include "ops/elementwise/funcs.hpp"
#include "ops/gemm/cublaslt.hpp"

#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char const *argv[]) {

  Tensor a = empty({2, 2}, DataType::FLOAT32);
  Tensor b = empty({3, 2}, DataType::FLOAT32);
  Tensor c = empty({2, 2}, DataType::FLOAT32);

  ops::rand(a, 0);
  ops::rand(b, 1);

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "c (before): " << c << std::endl;

  //   ops::add(c, a, b);
  ops::gemm(c, a, b);

  std::cout << "c (after): " << c << std::endl;

  return 0;
}
