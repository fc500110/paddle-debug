// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference data reader

#include "data/reader.h"
#include <paddle_inference_api.h>
#include <algorithm>
#include <boost/endian/conversion.hpp>
#include <functional>

namespace paddle {
namespace debug {
Reader::Reader(const std::string &filename)
    : file_(filename, std::ios::binary) {
  Init();
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

void Reader::ResetBatchData() {
  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i].lod.clear();
    inputs_[i].emplace_back(levels_[i], {0});
  }
}

bool Reader::NextBatch() {
  ResetBatchData();

  struct DataBuf {
    // std::vector<char> data;
    std::string data;
    std::vector<std::vector<size_t>> lod;
  };

  std::vector<DataBuf> buffer(inputs_.size());

  for (int i = 0; i < buffer.size(); i++) {
    buffer[i].lod.assign(levels_[i], {0});
  }

  size_t data_byte;
  if (inputs_[i].dtype == PaddleDType::INT32) {
    data_byte = sizeof(int32_t);
  } else if (inputs_[i].dtype == PaddleDType::INT64) {
    data_byte = sizeof(int64_t);
  } else if (inputs_[i].dtype == PaddleDType::FLOAT32) {
    data_byte = sizeof(float);
  }

  std::vector<std::vector<size_t>> lod;
  for (int i = 0; i < batch_size_; i++) {
    for (int j = 0; j < inputs_.size(); j++) {
      // LoD
      if (levels_[j] != 0) {
        Read(&lod);
        std::copy(lod.begin(), lod.end(), std::back_inserter(buffer[j].lod));
      }

      // raw data
      int length;
      Read(&length);
      std::copy_n(std::istream_iterator<char>(file_), length * data_byte,
                  std::back_inserter(buffer[j].data));
    }
  }

  for (int i = 0; i < inputs_.size(); i++) {
    auto *tensor = inputs_[i];
    const auto &data_buf = buffer[i];
    tensor->data.Resize(data_buf.data.size());
    std::copy(data_buf.data.begin(), data_buf.data.end(),
              static_cast<char *>(tensor->data.data()));

    for (auto &item : data_buf.lod) {
      std::partial_sum(item.begin(), item.end(), item.begin(),
                       std::plus<size_t>());
    }
  }
}

void GetPaddleDType(const std::string &name, PaddleDType *dtype) {
  if (name == "FLOAT32") {
    *dtype = PaddleDType::FLOAT32;
  } else if (name == "INT32") {
    *dtype = PaddleDType::INT32;
  } else if (name == "INT64") {
    *dtype = PaddleDType::INT64;
  } else {
    dtype = nullptr;
  }
}
}  // namespace debug
}  // namespace paddle
