#pragma once

#include "macros.hpp"

#include <cstddef>
#include <initializer_list>
#include <string>

template <typename T>
struct ConstArray {
  static const size_t MAX_SIZE = 8;
  T data[MAX_SIZE];
  size_t size;

  HOST_DEVICE ConstArray() : size(0) {}

  HOST_DEVICE ConstArray(std::initializer_list<T> list) : size(list.size()) {
    size_t i = 0;
    for (T value : list) {
      data[i++] = value;
    }
  }

  HOST_DEVICE ConstArray(size_t size, T value) : size(size) {
    for (size_t i = 0; i < size; i++) {
      data[i] = value;
    }
  }

  HOST_DEVICE void push_back(T value) { data[size++] = value; }

  HOST_DEVICE const T *begin() const { return data; }
  HOST_DEVICE const T *end() const { return data + size; }
  HOST_DEVICE T operator[](size_t i) const { return data[i]; }
  HOST_DEVICE T &operator[](size_t i) { return data[i]; }
  HOST_DEVICE bool operator==(const ConstArray &other) const {
    if (size != other.size) {
      return false;
    }
    for (size_t i = 0; i < size; i++) {
      if (data[i] != other.data[i]) {
        return false;
      }
    }
    return true;
  }
  HOST_DEVICE bool operator!=(const ConstArray &other) const {
    return !(*this == other);
  }
  std::string to_string() const {
    std::string result = "(";
    for (size_t i = 0; i < size; i++) {
      result += std::to_string(data[i]);
      if (i < size - 1) {
        result += ", ";
      }
    }
    result += ")";
    return result;
  }
};