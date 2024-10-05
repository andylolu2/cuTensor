#pragma once

#include "core/ConstArray.cuh"
#include "macros.hpp"

using SizeArray = ConstArray<size_t>;

HOST_DEVICE size_t prod(const SizeArray &array);