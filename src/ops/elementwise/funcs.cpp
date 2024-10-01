#include "ops/elementwise/funcs.hpp"

#include "core/Tensor.hpp"
#include "ops/elementwise/host.hpp"
#include "ops/utils.cuh"

#include <curand_kernel.h>

namespace ops {

namespace detail {
struct Add {
  template <typename T>
  DEVICE T call(Coord coord, T a, T b) const {
    return a + b;
  }
};
} // namespace detail

void add(Tensor output, Tensor a, Tensor b) {
  elementwise(detail::Add{}, output, a, b);
}

namespace detail {
struct Fill {
  float value;

  Fill(float value) : value(value) {}

  template <typename T>
  DEVICE T call(Coord coord) const {
    return static_cast<T>(value);
  }
};
} // namespace detail

void fill(Tensor output, float value) {
  elementwise(detail::Fill{value}, output);
}

namespace detail {
struct Rand {
  size_t seed;
  Rand(size_t seed) : seed(seed) {}

  template <typename T>
  DEVICE T call(Coord coord) const {
    auto tid = global_thread_index();
    curandState state;
    curand_init(seed, tid, 0, &state);
    return static_cast<T>(curand_normal(&state));
  }
};
} // namespace detail

void rand(Tensor output, size_t seed) {
  elementwise(detail::Rand{seed}, output);
}

} // namespace ops
