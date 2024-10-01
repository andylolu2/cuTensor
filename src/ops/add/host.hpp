#pragma once

// #include "core/ConstArray.cuh"
#include "core/Tensor.hpp"

// template <typename T>
// void add(
//     T *a, ConstArray a_shape, ConstArray a_strides, T *b, ConstArray b_shape,
//     ConstArray b_strides, T *c, ConstArray c_shape, ConstArray c_strides
// );

void add_tensors(Tensor output, Tensor a, Tensor b);