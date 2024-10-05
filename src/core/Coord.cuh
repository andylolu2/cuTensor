#pragma once

#include "core/SizeArray.cuh"
#include "macros.hpp"

#include <string>

class Coord {
private:
  SizeArray coord;
  SizeArray domain;

public:
  HOST_DEVICE Coord(SizeArray coord, SizeArray domain);
  HOST_DEVICE ~Coord();
  HOST_DEVICE size_t n_dims() const;
  HOST_DEVICE size_t operator[](size_t i) const;
  std::string to_string() const;
};

HOST_DEVICE Coord from_1d(size_t index_1d, SizeArray domain);

HOST_DEVICE size_t to_offset(const Coord &coord, const SizeArray &strides);