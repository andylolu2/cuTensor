#include "ops/add/host.hpp"

#include "ops/add/kernel.cuh"

template <typename T>
void add(
    T *a, ConstArray a_shape, ConstArray a_strides, T *b, ConstArray b_shape,
    ConstArray b_strides, T *c, ConstArray c_shape, ConstArray c_strides
) {
  auto [grid_dim, block_dim] = launch_config_1d(prod(c_shape));
  add_kernel<T><<<grid_dim, block_dim>>>(
      a, a_shape, a_strides, b, b_shape, b_strides, c, c_shape, c_strides
  );
}

void add_tensors(Tensor a, Tensor b, Tensor c) {
  if (a.getDtype() != b.getDtype() || a.getDtype() != c.getDtype()) {
    throw std::runtime_error(fmt::format(
        "The data types of the input arrays do not match: {} != {} != {}",
        a.getDtype().name, b.getDtype().name, c.getDtype().name
    ));
  }
  if (a.getDims() != b.getDims() || a.getDims() != c.getDims()) {
    throw std::runtime_error(fmt::format(
        "The sizes of the input arrays do not match: {} != {} != {}",
        a.getDims().to_string(), b.getDims().to_string(),
        c.getDims().to_string()
    ));
  }

  SWITCH_DATATYPE(a.getDtype(), ([&] {
                    add(reinterpret_cast<T *>(a.getData()), a.getDims(),
                        a.getStrides(), reinterpret_cast<T *>(b.getData()),
                        b.getDims(), b.getStrides(),
                        reinterpret_cast<T *>(c.getData()), c.getDims(),
                        c.getStrides());
                  }));
}