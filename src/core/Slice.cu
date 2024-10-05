#include "core/Slice.cuh"

Slice::Slice(SliceArray slices, SizeArray domain)
    : slices(slices), domain(domain) {}

Slice::~Slice() {}

size_t Slice::n_dims() const { return slices.size; }

Slice::SliceT Slice::operator[](size_t i) const { return slices[i]; }

Coord Slice::coord_from_1d(size_t index_1d) const {
  // The index_1d enumerates the ":"-ed into dimensions.
  // Get the ":"-ed dimensions.
  SizeArray colon_dims;
  for (size_t i = 0; i < slices.size; i++) {
    if (!slices[i].has_value()) {
      colon_dims.push_back(domain[i]);
    }
  }
  // Get the coord in the  ":"-ed dimensions.
  Coord colon_coord = from_1d(index_1d, colon_dims);
  // Expand the coord to the full domain.
  SizeArray full_coord;
  size_t colon_coord_index = 0;
  for (size_t i = 0; i < slices.size; i++) {
    if (slices[i].has_value()) {
      full_coord.push_back(slices[i].value());
    } else {
      full_coord.push_back(colon_coord[colon_coord_index]);
      colon_coord_index++;
    }
  }
  return Coord(full_coord, domain);
}

std::string Slice::to_string() const {
  std::string result = "[";
  for (size_t i = 0; i < slices.size; i++) {
    if (i > 0) {
      result += ", ";
    }
    SliceT slice = slices[i];
    if (slice.has_value()) {
      result += std::to_string(slice.value());
    } else {
      result += ":";
    }
  }
  result += "]";
  return result;
}