#pragma once

#include "common/cuda_utils.hpp"
#include "core/ConstArray.cuh"

#include <string>

class Coord {
private:
  ConstArray coord;

public:
  HOST_DEVICE Coord(ConstArray coord);
  HOST_DEVICE ~Coord();
  HOST_DEVICE size_t n_dims() const;
  HOST_DEVICE size_t operator[](size_t i) const;
  std::string to_string() const;
};

HOST_DEVICE Coord from_1d(size_t index_1d, ConstArray dims);

HOST_DEVICE size_t to_offset(const Coord &coord, const ConstArray &strides);