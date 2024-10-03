#pragma once

#include "macros.hpp"

#include <cstddef>
#include <initializer_list>
#include <string>

struct ConstArray {
  static const size_t MAX_SIZE = 8;
  size_t data[MAX_SIZE];
  size_t size;

  HOST_DEVICE ConstArray(std::initializer_list<size_t> list);
  HOST_DEVICE ConstArray(size_t size, size_t value);

  HOST_DEVICE const size_t *begin() const;
  HOST_DEVICE const size_t *end() const;
  HOST_DEVICE size_t operator[](size_t i) const;
  HOST_DEVICE size_t &operator[](size_t i);
  HOST_DEVICE bool operator==(const ConstArray &other) const;
  HOST_DEVICE bool operator!=(const ConstArray &other) const;
  std::string to_string() const;
};

HOST_DEVICE size_t prod(const ConstArray &array);