#pragma once

#include "core/Slice.cuh"
#include "core/Tensor.hpp"

namespace ops {

Tensor view(Tensor tensor, Slice::SliceArray slice);

}