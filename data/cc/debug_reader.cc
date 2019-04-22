// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Change Fu (fuchang01@baidu.com)
//
// Paddle inference data reader

#include "debug_reader.h"
#include <paddle_inference_api.h>
#include <algorithm>
#include <boost/endian/conversion.hpp>
#include <boost/make_unique.hpp>
#include <functional>
#include <iterator>

namespace paddle {
namespace debug {

Reader::Reader(std::unique_ptr<std::istream> stream)
    : stream_(std::move(stream)) {
  Init();
}

Reader::Reader(const std::string &filename)
    : Reader(boost::make_unique<std::fstream>(filename, std::ios::binary)) {}

void Reader::Init() {
  size_t num_inputs{0};
  Get(&num_inputs);

  inputs_.resize(num_inputs);
  levels_.resize(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    PaddleTensor *tensor = &inputs_[i];
    Get(&tensor->name);
    Get(&tensor->shape);
    std::string dtype_name;
    Get(&dtype_name);
    GetPaddleDType(dtype_name, &tensor->dtype);
    Get(&levels_[i]);
  }
}

void Reader::ResetBatchData() noexcept {
  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i].lod.clear();
    inputs_[i].lod = {static_cast<size_t>(levels_[i]),
                      std::vector<size_t>({0})};
  }
}

bool Reader::NextBatch() {
  ResetBatchData();

  struct DataBuf {
    std::string data;
    std::vector<std::vector<size_t>> lod;
  };

  std::vector<DataBuf> buffer(inputs_.size());

  for (int i = 0; i < buffer.size(); i++) {
    buffer[i].lod.assign(levels_[i], {0});
  }

  auto get_data_byte = [](const PaddleTensor &tensor) {
    if (tensor.dtype == PaddleDType::INT64) return sizeof(int64_t);
    if (tensor.dtype == PaddleDType::FLOAT32) return sizeof(float);
  };

  std::vector<std::vector<size_t>> lod;
  for (int i = 0; i < batch_size_; i++) {
    for (int j = 0; j < inputs_.size(); j++) {
      // LoD
      if (levels_[j] != 0) {
        Get(&lod);
        std::copy(lod.begin(), lod.end(), std::back_inserter(buffer[j].lod));
      }

      // raw data
      size_t length;
      Get(&length);
      std::copy_n(std::istream_iterator<char>(*stream_),
                  length * get_data_byte(inputs_[j]),
                  std::back_inserter(buffer[j].data));
    }
  }

  for (int i = 0; i < inputs_.size(); i++) {
    auto *tensor = &inputs_[i];
    auto &data_buf = buffer[i];
    tensor->data.Resize(data_buf.data.size());
    std::copy(data_buf.data.begin(), data_buf.data.end(),
              static_cast<char *>(tensor->data.data()));

    auto size = tensor->data.length() / get_data_byte(inputs_[i]);
    if (tensor->dtype == PaddleDType::FLOAT32) {
      auto *data = static_cast<float *>(tensor->data.data());
      transform_big_to_native(data, data + size);
    } else if (tensor->dtype == PaddleDType::INT64) {
      auto *data = static_cast<int64_t *>(tensor->data.data());
      transform_big_to_native(data, data + size);
    }

    // for (auto it = data_buf.lod.begin(); it != data_buf.lod.end(); ++it) {
    //  std::partial_sum(it->begin(), it->end(), it->begin(),
    //                   std::plus<size_t>());
    //}
    for (auto &item : data_buf.lod) {
      std::partial_sum(item.begin(), item.end(), item.begin(),
                       std::plus<size_t>());
    }
  }
}

void GetPaddleDType(const std::string &name, PaddleDType *dtype) {
  if (name == "FLOAT32") {
    *dtype = PaddleDType::FLOAT32;
  } else if (name == "INT64") {
    *dtype = PaddleDType::INT64;
  } else {
    dtype = nullptr;
  }
}
}  // namespace debug
}  // namespace paddle
