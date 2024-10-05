#pragma once

#include "core/ConstArray.cuh"
#include "core/Coord.cuh"
#include "core/SizeArray.cuh"
#include "macros.hpp"

#include <cuda/std/optional>
#include <string>

class Slice {
public:
  using SliceT = std::optional<size_t>; // None = ":" in NumPy
  using SliceArray = ConstArray<SliceT>;

private:
  SliceArray slices;
  SizeArray domain;

public:
  HOST_DEVICE Slice(SliceArray slices, SizeArray domain);
  HOST_DEVICE ~Slice();
  HOST_DEVICE size_t n_dims() const;
  HOST_DEVICE SliceT operator[](size_t i) const;
  HOST_DEVICE Coord coord_from_1d(size_t index_1d) const;
  std::string to_string() const;
};