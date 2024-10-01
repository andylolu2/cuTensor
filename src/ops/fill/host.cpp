// #include "ops/fill/host.hpp"

// #include "ops/fill/kernel.cuh"
// #include "ops/utils.cuh"
#include "ops/elementwise/host.hpp"

// template <typename T>
// void fill(T value, T *a, ConstArray a_shape, ConstArray a_strides) {
//   auto [grid_dim, block_dim] = launch_config_1d(prod(a_shape));
//   fill_kernel<T><<<grid_dim, block_dim>>>(value, a, a_shape, a_strides);
// }

// void fill_tensor(Tensor a, float value) {
//   SWITCH_DATATYPE(a.getDtype(), ([&] {
//                     fill<T>(
//                         static_cast<T>(value),
//                         reinterpret_cast<T *>(a.getData()), a.getDims(),
//                         a.getStrides()
//                     );
//                   }));
// }

struct Fill {
  float value;

  Fill(float value) : value(value) {}

  template <typename T>
  DEVICE T operator()() const {
    return static_cast<T>(value);
  }
};

void fill_tensor(Tensor a, float value) { elementwise_tensor(Fill{value}, a); }