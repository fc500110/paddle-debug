// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference data reader

#pragma once

#include <paddle_inference_api.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "paddle_debug_helper.h"

namespace paddle {
namespace debug {

using paddle::PaddleBuf;
using paddle::PaddleDType;
using paddle::PaddleTensor;

void GetPaddleDType(const std::string &name, PaddleDType *dtype);

class Reader {
 public:
  explicit Reader(const std::string &filename);
  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;

  void Init();
  void SetBatchSize(int batch_size) noexcept { batch_size_ = batch_size; }
  bool NextBatch();
  const std::vector<PaddleTensor> &data() const noexcept { return inputs_; }

 private:
  template <typename T>
  void Get(T *data) {
    Read(&stream_, data);
  }

  void ResetBatchData() noexcept;

  int batch_size_{1};

  std::ifstream stream_;

  std::vector<PaddleTensor> inputs_;
  std::vector<int> levels_;
};

}  // namespace debug
}  // namespace paddle
