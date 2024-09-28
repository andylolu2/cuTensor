#include "detail/Index.cuh"

__host__ __device__ Index::Index(std::vector<size_t> coord) : coord(coord) {}

// __host__ __device__ Index
// Index::from_1d(size_t index_1d, std::vector<size_t> dims) {
//   std::vector<size_t> coord(dims.size());
//   size_t index_1d_ = index_1d;
//   for (size_t i = dims.size() - 1; i > 0; i--) {
//     coord[i] = index_1d_ % dims[i];
//     index_1d_ /= dims[i];
//   }
//   return Index(coord);
// }

__host__ __device__ Index::~Index() {}

__host__ __device__ size_t Index::n_dims() const { return coord.size(); }

__host__ __device__ size_t Index::operator[](size_t i) const {
  return coord[i];
}