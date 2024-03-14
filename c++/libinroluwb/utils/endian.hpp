#pragma once

#include <glog/logging.h>

#include <iostream>
#include <algorithm>

inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

namespace inrol {
  template <typename T>
  T ReverseBytes(const T& data) {
    T data_reversed = data;
    std::reverse(
      reinterpret_cast<char*>(&data_reversed),
      reinterpret_cast<char*>(&data_reversed) + sizeof(T));
    return data_reversed;
  }

  template <typename T>
  T LittleEndianToNative(const T x) {
    if (IsLittleEndian()) {
      return x;
    } else {
      return ReverseBytes(x);
    }
  }

  template <typename T>
  T readBinaryLittleEndian(std::istream* stream) {
    T data_little_endian;
    stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
    return LittleEndianToNative(data_little_endian);
  }
}  // namespace inrol
