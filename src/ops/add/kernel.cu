#pragma once

#include "common/cuda_utils.cuh"
#include "detail/Index.hpp"
#include "detail/RuntimeArray.hpp"
#include "fmt/format.h"

#include <vector>

template <typename T>
__global__ void add_kernel(RuntimeArray a, RuntimeArray b, RuntimeArray c) {
  auto tid = global_thread_index();
  if (tid < c.size()) {
    Index index = Index::from_1d(tid, c.size());
    c[index] = a[index] + b[index];
  }
}

template <typename T>
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
        a.dtype(), b.dtype(), c.dtype()
    ));
  }

  auto [grid_dim, block_dim] = launch_config_1d(c.size());
  SWITCH_DATATYPE(a.dtype(), ([&] {
                    add_kernel<T><<<grid_dim, block_dim>>>(a, b, c);
                  }));
}