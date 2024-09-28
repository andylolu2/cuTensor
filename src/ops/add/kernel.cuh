#pragma once

#include "common/cuda_utils.cuh"
#include "detail/Index.cuh"
#include "detail/RuntimeArray.cuh"
#include "fmt/format.h"

#include <vector>

template <typename T>
__global__ void add_kernel(RuntimeArray a, RuntimeArray b, RuntimeArray c) {
  auto tid = global_thread_index();
  if (tid < c.size()) {
    Index index = Index::from_1d(tid, c.shape());
    T a_value = *reinterpret_cast<T *>(a(index));
    T b_value = *reinterpret_cast<T *>(b(index));
    T *c_ptr = reinterpret_cast<T *>(c(index));
    *c_ptr = a_value + b_value;
  }
}

void add(RuntimeArray a, RuntimeArray b, RuntimeArray c) {
  if (a.size() != b.size() || a.size() != c.size()) {
    throw std::runtime_error(fmt::format(
        "The sizes of the input arrays do not match: {} != {} != {}", a.size(),
        b.size(), c.size()
    ));
  }
  if (a.dtype() != b.dtype() || a.dtype() != c.dtype()) {
    throw std::runtime_error(fmt::format(
        "The data types of the input arrays do not match: {} != {} != {}",
        dtype_name(a.dtype()), dtype_name(b.dtype()), dtype_name(c.dtype())
    ));
  }

  SWITCH_DATATYPE(a.dtype(), ([&] {
                    auto [grid_dim, block_dim] = launch_config_1d(c.size());
                    add_kernel<T><<<grid_dim, block_dim>>>(a, b, c);
                  }));
}