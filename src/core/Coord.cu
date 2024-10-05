#include "core/Coord.cuh"

#include <cassert>

HOST_DEVICE Coord::Coord(SizeArray coord, SizeArray domain)
    : coord(coord), domain(domain) {}

HOST_DEVICE Coord::~Coord() {}

HOST_DEVICE size_t Coord::n_dims() const { return coord.size; }

HOST_DEVICE size_t Coord::operator[](size_t i) const { return coord[i]; }

std::string Coord::to_string() const { return coord.to_string(); }

HOST_DEVICE Coord from_1d(size_t index_1d, SizeArray domain) {
  SizeArray coord(domain);
  for (size_t i = domain.size; i-- > 0;) {
    coord[i] = index_1d % domain[i];
    index_1d /= domain[i];
  }
  return Coord(coord, domain);
};

HOST_DEVICE size_t to_offset(const Coord &coord, const SizeArray &strides) {
  assert(coord.n_dims() == strides.size);
  size_t offset = 0;
  for (size_t i = 0; i < coord.n_dims(); i++) {
    offset += coord[i] * strides[i];
  }
  return offset;
}