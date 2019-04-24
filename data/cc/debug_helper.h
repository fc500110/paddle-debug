#pragma once

#include <algorithm>
#include <boost/endian/conversion.hpp>
#include <istream>
#include <type_traits>

namespace paddle {
namespace debug {

using boost::endian::big_to_native;
using boost::endian::big_to_native_inplace;

template <typename Iter>
void transform_big_to_native(Iter begin, Iter end) {
  using value_t = typename std::iterator_traits<Iter>::value_type;
  std::for_each(begin, end, &big_to_native_inplace<value_t>);
}

template <>
void transform_big_to_native(float *begin, float *end) {
  constexpr bool is_big_endian =
      (boost::endian::order::native == boost::endian::order::big);
  if (is_big_endian) return;

  auto float_conv = [](float &v) {
    union {
      float value;
      char c[sizeof(float)];
    } tmp;
    tmp.value = v;
    std::swap(tmp.c[0], tmp.c[3]);
    std::swap(tmp.c[1], tmp.c[2]);
    // std::reverse(std::begin(tmp.c), std::end(tmp.c));
    v = tmp.value;
  };

  // int32_t *i_begin = reinterpret_cast<int32_t *>(begin);
  // int32_t *i_end = reinterpret_cast<int32_t *>(end);
  std::for_each(begin, end, float_conv);
}

template <typename T>
void Read(std::istream *stream, T *data) {
  stream->read(reinterpret_cast<char *>(data), sizeof(T));
  *data = big_to_native(*data);
}

template <>
void Read(std::istream *stream, float *data) {
  char *buffer = reinterpret_cast<char *>(data);
  stream->read(buffer, sizeof(float));
  /*if (boost::endian::order::native == boost::endian::order::big) {
    std::swap(buffer[0], buffer[3]);
    std::swap(buffer[1], buffer[2]);
  }*/
};

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
