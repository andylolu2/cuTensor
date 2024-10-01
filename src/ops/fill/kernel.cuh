#pragma once

#include "core/ConstArray.cuh"
#include "core/Coord.cuh"
#include "ops/utils.cuh"

template <typename T>
__global__ void
fill_kernel(T value, T *a, ConstArray a_shape, ConstArray a_strides) {
  auto tid = global_thread_index();
  if (tid < prod(a_shape)) {
    auto coord = from_1d(tid, a_shape);
    auto a_offset = to_offset(coord, a_strides);
    a[a_offset] = value;
  }
}