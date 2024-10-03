#pragma one

#include <stddef.h>

class DeviceArray {
private:
  void *data;
  size_t size;

public:
  DeviceArray(size_t size);
  ~DeviceArray();

  void *getData() const;
  size_t getSize() const;
};