#pragma once

#include "ConstArray.cuh"
#include "detail/Coord.cuh"
#include "ops/utils.cuh"

template <typename T>
__global__ void add_kernel(
    T *a, ConstArray a_shape, ConstArray a_strides, T *b, ConstArray b_shape,
    ConstArray b_strides, T *c, ConstArray c_shape, ConstArray c_strides
) {
  auto tid = global_thread_index();
  if (tid < prod(c_shape)) {
    auto coord = from_1d(tid, c_shape);
    auto a_offset = to_offset(coord, a_strides);
    auto b_offset = to_offset(coord, b_strides);
    auto c_offset = to_offset(coord, c_strides);
    c[c_offset] = a[a_offset] + b[b_offset];
  }
}