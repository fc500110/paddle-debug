// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference data reader

#pragma once

#include <paddle_inference_api.h>
#include <boost/endian/conversion.hpp>
#include <algorithm>
#include <fsteram>
#include <string>
#include <vector>

namespace paddle {
namespace debug {

using paddle::PaddleBuf;
using paddle::PaddleDType;
using paddle::PaddleTensor;

static void GetPaddleDType(const std::string &name, PaddleDType *dtype);

class Reader {
 public:
  explicit Reader(const std::string &filename);
  explicit Reader(std::istream &&in);
  Reader() = delete;
  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;

  void Init();
  void SetBatchSize(int batch_size) noexcept { batch_size_ = batch_size; }
  bool NextBatch();
  const std::vector<PaddleTensor> &batch() const noexcept { return inputs_; }

 private:
  template <typename T>
  void Read(T *data);

  void ResetBatchData() noexcept;

  std::ifstream file_;
  int batch_size_{1};

  std::vector<PaddleTensor> inputs_;
  std::vector<int> levels_;
};

template <typename T>
void Reader::Read(T *data) {
  file_.read(reinterpret_cast<char *>(data), sizeof(T));
  *data = boost::endian::big_to_native(*data);
}

template <>
void Reader::Read(std::string *data) {
  int size{0};
  Read(&size);
  data->assign(size, '\0');
  file_.read(&data->front(), size);
}

template <typename T>
void Reader::Read(std::vector<T> *data) {
  size_t size{0};
  Read(&size);

  data->assign(size, T{});
  for (auto &v : *data) {
    Read<T>(&v);
  }
}

template <typename T>
void Reader::Read(std::vector<std::vector<T>> *data) {
  size_t size{0};
  Read(&size);
  data->assign(size, std::vector<T>());
  for (auto &v : *data) {
    Read(&v);
  }
}

template <>
void Reader::Read(PaddleDType *data) {
  std::string name;
  Read(&name);
  GetPaddleDType(name, data);
}

void Reader::Init() {
  int num_inputs{0};
  Read(&num_inputs);

  inputs_.resize(num_inputs);
  levels_.resize(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    PaddleTensor *tensor = inputs_[i];
    Read(&tensor->name);
    Read(&tensor->shape);
    Read(&tensor->dtype);
    Read(&levels_[i]);
  }
}

template <typename Iter>
void big_to_native(Iter begin, Iter end) {
  using value_type = std::iterator_trits<Iter>::value_type;
  std::transform(begin, end, begin, boost::endian::beg_to_native<value_type>);
}

}  // namespace debug
}  // namespace paddle
