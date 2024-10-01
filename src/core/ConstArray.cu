#include "core/ConstArray.cuh"

HOST_DEVICE ConstArray::ConstArray(std::initializer_list<size_t> list)
    : size(list.size()) {
  size_t i = 0;
  for (size_t value : list) {
    data[i++] = value;
  }
}

HOST_DEVICE size_t ConstArray::operator[](size_t i) const { return data[i]; }
HOST_DEVICE size_t &ConstArray::operator[](size_t i) { return data[i]; }

HOST_DEVICE bool ConstArray::operator==(const ConstArray &other) const {
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

HOST_DEVICE bool ConstArray::operator!=(const ConstArray &other) const {
  return !(*this == other);
}

std::string ConstArray::to_string() const {
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

HOST_DEVICE size_t prod(const ConstArray &array) {
  size_t result = 1;
  for (size_t i = 0; i < array.size; i++) {
    result *= array[i];
  }
  return result;
}