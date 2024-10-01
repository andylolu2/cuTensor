#pragma once

// #include "core/ConstArray.cuh"
#include "core/Tensor.hpp"

// template <typename T>
// void fill(T value, T *a, ConstArray a_shape, ConstArray a_strides);

void fill_tensor(Tensor a, float value);