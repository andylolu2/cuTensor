#pragma once

#include "core/Tensor.hpp"

namespace ops {

void add(Tensor output, Tensor a, Tensor b);

void fill(Tensor output, float value);

void rand(Tensor output, size_t seed);

} // namespace ops
