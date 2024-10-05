#pragma once

#include "core/DataType.hpp"
#include "core/SizeArray.cuh"
#include "core/Tensor.hpp"

namespace ops {

Tensor empty(SizeArray dims, const DataType &dtype);

}