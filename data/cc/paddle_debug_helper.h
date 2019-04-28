// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference debug helper

#pragma once

#include <arpa/inet.h>
#include <algorithm>
#include <istream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace paddle {
namespace debug {

class Endian {
 private:
  Endian() = delete;
  static constexpr uint32_t uint32_v_{0x01020304};
  static constexpr uint8_t magic_v_ = static_cast<const uint8_t &>(uint32_v_);
  static constexpr bool is_big_ = magic_v_ == 0x01;
  static constexpr bool is_little_ = magic_v_ == 0x04;

 public:
  enum class Order { kBig, kLittle, kNative = is_big_ ? kBig : kLittle };

  template <typename T>
  static void Reverse(T &v) {
    union {
      T value;
      char c[sizeof(T)];
    } tmp;
    tmp.value = v;
    std::reverse(tmp.c, tmp.c + sizeof(T));
    v = tmp.value;
  }

  template <typename T>
  static void BigToNativeInplace(T &v) {
    if (is_big_) return;
    Reverse(v);
  }

  template <typename T>
  static void NativeToBigInplace(T &v) {
    if (is_big_) return;
    Reverse(v);
  }
};

template <typename T>
void Read(std::istream *stream, T *data) {
  stream->read(reinterpret_cast<char *>(data), sizeof(T));
  Endian::BigToNativeInplace(*data);
}

template <>
void Read(std::istream *stream, float *data) {
  char *buffer = reinterpret_cast<char *>(data);
  stream->read(buffer, sizeof(float));
}

template <>
void Read(std::istream *stream, std::string *data) {
  int size{0};
  Read(stream, &size);
  data->assign(size, '\0');
  stream->read(&data->front(), size);
}

template <typename T>
void Read(std::istream *stream, std::vector<T> *data) {
  size_t size{0};
  Read(stream, &size);
  data->clear();
  data->assign(size, T());
  for (auto &v : *data) {
    Read(stream, &v);
  }
}

template <typename T>
void Read(std::istream *stream, std::vector<std::vector<T>> *data) {
  size_t size{0};
  Read(stream, &size);
  data->clear();
  data->assign(size, std::vector<T>());
  for (auto &v : *data) {
    Read(stream, &v);
  }
}

}  // namespace debug
}  // namespace paddle
