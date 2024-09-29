#include "ops/fill/host.hpp"

#include "ops/fill/kernel.cuh"

template <typename T>
void fill(T value, T *a, ConstArray a_shape, ConstArray a_strides) {
  auto [grid_dim, block_dim] = launch_config_1d(prod(a_shape));
  fill_kernel<T><<<grid_dim, block_dim>>>(value, a, a_shape, a_strides);
}

void fill_tensor(Tensor a, float value) {
  SWITCH_DATATYPE(a.getDtype(), ([&] {
                    fill<T>(
                        static_cast<T>(value),
                        reinterpret_cast<T *>(a.getData()), a.getDims(),
                        a.getStrides()
                    );
                  }));
}