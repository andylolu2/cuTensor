#pragma once

#include "core/ConstArray.cuh"
#include "core/Coord.cuh"
#include "ops/utils.cuh"

#include <curand_kernel.h>
#include <initializer_list>

struct TensorDescriptor {
  void *data;
  ConstArray shape;
  ConstArray strides;
};

template <typename T, typename ElementWiseFunc, typename... TensorDescriptors>
__global__ void elementwise_kernel(
    ElementWiseFunc func, TensorDescriptor output, TensorDescriptors... tensors
) {
  auto tid = global_thread_index();
  if (tid < prod(output.shape)) {
    auto coord = from_1d(tid, output.shape);
    auto output_offset = to_offset(coord, output.strides);
    auto output_data = reinterpret_cast<T *>(output.data);
    output_data[output_offset] = func.template call<T>(
        coord, reinterpret_cast<T *>(tensors.data
               )[to_offset(coord, tensors.strides)]...
    );
  }
}