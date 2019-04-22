// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference data reader

#pragma once

#include <paddle_inference_api.h>
#include <algorithm>
#include <boost/endian/conversion.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "debug_helper.h"

namespace paddle {
namespace debug {

using paddle::PaddleBuf;
using paddle::PaddleDType;
using paddle::PaddleTensor;

void GetPaddleDType(const std::string &name, PaddleDType *dtype);

class Reader {
 public:
  explicit Reader(std::unique_ptr<std::istream> stream);
  explicit Reader(const std::string &filename);
  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;

  void Init();
  void SetBatchSize(int batch_size) noexcept { batch_size_ = batch_size; }
  bool NextBatch();
  const std::vector<PaddleTensor> &batch() const noexcept { return inputs_; }

 private:
  Reader();
  class Impl;

  template <typename T>
  void Get(T *data) {
    Read(stream_.get(), data);
  }

  // template <>
  // void Read(std::string *data);

  // template <typename T>
  // void Read(std::vector<T> *);

  // template <typename T>
  // void Read(std::vector<std::vector<T>> *);

  void ResetBatchData() noexcept;

  int batch_size_{1};

  // std::istream stream_;
  std::unique_ptr<std::istream> stream_;

  std::vector<PaddleTensor> inputs_;
  std::vector<int> levels_;
};

// template <typename T>
// void Reader::Read(T *data) {
//  file_.read(reinterpret_cast<char *>(data), sizeof(T));
//  *data = boost::endian::big_to_native(*data);
//}
//
// template <>
// void Reader::Read(std::string *data) {
//  int size{0};
//  Read(&size);
//  data->assign(size, '\0');
//  file_.read(&data->front(), size);
//}
//
// template <typename T>
// void Reader::Read(std::vector<T> *data) {
//  size_t size{0};
//  Read(&size);
//
//  data->assign(size, T{});
//  for (auto &v : *data) {
//    Read<T>(&v);
//  }
//}
//
// template <typename T>
// void Reader::Read(std::vector<std::vector<T>> *data) {
//  size_t size{0};
//  Read(&size);
//  data->assign(size, std::vector<T>());
//  for (auto &v : *data) {
//    Read<std::vector<T>>(&v);
//  }
//}
//
// template <>
// void Reader::Read(PaddleDType *data) {
//  std::string name;
//  Read(&name);
//  GetPaddleDType(name, data);
//}

}  // namespace debug
}  // namespace paddle
