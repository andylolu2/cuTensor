#pragma once

#include "core/ConstArray.cuh"
#include "core/Tensor.hpp"
#include "ops/elementwise/kernel.cuh"

template <typename T, typename ElementWiseFunc, typename... TensorDescriptors>
void elementwise(
    ElementWiseFunc func, TensorDescriptor output, TensorDescriptors... tensors
) {
  auto [grid_dim, block_dim] = launch_config_1d(prod(output.shape));
  elementwise_kernel<T, ElementWiseFunc, TensorDescriptors...>
      <<<grid_dim, block_dim>>>(func, output, tensors...);
}

template <typename ElementWiseFunc, typename... Tensors>
void elementwise_tensor(
    ElementWiseFunc func, Tensor output, Tensors... tensors
) {
  SWITCH_DATATYPE(
      output.getDtype(), ([&] {
        elementwise<T>(
            func,
            TensorDescriptor{
                output.getData(), output.getDims(), output.getStrides()
            },
            TensorDescriptor{
                tensors.getData(), tensors.getDims(), tensors.getStrides()
            }...
        );
      })
  );
}
