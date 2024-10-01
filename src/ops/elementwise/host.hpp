#pragma once

#include "core/ConstArray.cuh"
#include "core/Tensor.hpp"
#include "ops/elementwise/kernel.cuh"

#include <algorithm>
#include <stdexcept>

/// @brief Dispatch the elementwise kernel with appropriate configuration
template <typename T, typename ElementWiseFunc, typename... TensorDescriptors>
void elementwise_lower(
    ElementWiseFunc func, TensorDescriptor output, TensorDescriptors... tensors
) {
  auto [grid_dim, block_dim] = launch_config_1d(prod(output.shape));
  elementwise_kernel<T, ElementWiseFunc, TensorDescriptors...>
      <<<grid_dim, block_dim>>>(func, output, tensors...);
}

/// @brief Apply an elementwise operation to the given tensors
template <typename ElementWiseFunc, typename... Tensors>
void elementwise(ElementWiseFunc func, Tensor output, Tensors... tensors) {
  // Check all tensors have the same shape and dtype
  std::initializer_list<Tensor> inputs = {tensors...};
  if (!std::all_of(inputs.begin(), inputs.end(), [&](const Tensor &t) {
        return (t.getNDims() == output.getNDims()) &&
               (t.getDtype() == output.getDtype());
      })) {
    throw std::invalid_argument("All tensors must have the same shape and dtype"
    );
  }

  // Dispatch with correct datatype
  SWITCH_DATATYPE(
      output.getDtype(), ([&] {
        elementwise_lower<T>(
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
