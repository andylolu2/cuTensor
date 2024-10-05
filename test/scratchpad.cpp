#include "core/DataType.hpp"
#include "core/Tensor.hpp"
#include "macros.hpp"
#include "ops/create/empty.hpp"
#include "ops/create/view.hpp"
#include "ops/elementwise/funcs.hpp"
#include "ops/gemm/cublaslt.hpp"

#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char const *argv[]) {

  Tensor a = ops::empty({2, 2}, DataType::FLOAT32);
  Tensor b = ops::empty({2, 2}, DataType::FLOAT32);
  Tensor c = ops::empty({2, 2}, DataType::FLOAT32);

  ops::rand(a, 0);
  ops::rand(b, 1);

  std::cout << "a: " << a << std::endl;

  Tensor a_view = ops::view(a, {std::nullopt, 1});
  ops::fill(a_view, 10);

  std::cout << "a: " << a << std::endl;
  return 0;
  std::cout << "b: " << b << std::endl;
  std::cout << "c (before): " << c << std::endl;

  //   ops::add(c, a, b);
  ops::gemm(c, a, b);

  std::cout << "c (after): " << c << std::endl;

  return 0;
}
