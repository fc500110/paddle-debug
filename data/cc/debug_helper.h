#pragma once

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
  if (std::is_same<value_t, float>::value) {
    std::for_each(reinterpret_cast<int32_t *>(&(*begin)),
                  reinterpret_cast<int32_t *>(&(*end)),
                  &big_to_native_inplace<int32_t>);
  } else if (std::is_same<value_t, double>::value) {
    std::for_each(reinterpret_cast<int64_t *>(&(*begin)),
                  reinterpret_cast<int64_t *>(&(*end)),
                  &big_to_native_inplace<int64_t>);
  } else {
    std::for_each(begin, end, &big_to_native<value_t>);
  }
}

template <>
void transform_big_to_native(float *begin, float *end) {
  int32_t *i_begin = reinterpret_cast<int32_t *>(begin);
  int32_t *i_end = reinterpret_cast<int32_t *>(end);
  std::for_each(i_begin, i_end, &big_to_native_inplace<int32_t>);
}

template <typename T>
void Read(std::istream *stream, T *data) {
  stream->read(reinterpret_cast<char *>(data), sizeof(T));
  *data = big_to_native(*data);
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
  data->assign(size, T{});
  for (auto &v : *data) {
    Read(stream, &v);
  }
}

template <typename T>
void Read(std::istream *stream, std::vector<std::vector<T>> *data) {
  size_t size{0};
  Read(stream, &size);
  data->assign(size, std::vector<T>());
  for (auto &v : *data) {
    Read(stream, &v);
  }
}

}  // namespace debug
}  // namespace paddle
