#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include "debug_reader.h"

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

template <typename T>
void Write(T v, std::ostream *os) {
  auto bv = boost::endian::native_to_big(v);
  os->write(reinterpret_cast<char *>(&bv), sizeof(bv));
}

template <>
void Write(const std::string &v, std::ostream *os) {
  int size = v.size();
  Write(size, os);
  os->write(v.c_str(), size);
}

template <>
void Write(std::string v, std::ostream *os) {
  Write<const std::string &>(v, os);
}

template <>
void Write(float v, std::ostream *os) {
  char *c = reinterpret_cast<char *>(&v);
  if (boost::endian::order::native == boost::endian::order::little) {
    std::swap(c[0], c[3]);
    std::swap(c[1], c[2]);
  }
  os->write(c, sizeof(float));
}

template <typename T>
void Write(std::vector<T> v, std::ostream *os) {
  size_t size = v.size();
  boost::endian::native_to_big_inplace(size);
  os->write(reinterpret_cast<char *>(&size), sizeof(size));
  std::for_each(v.begin(), v.end(), [os](T &item) { Write(item, os); });
}

template <typename T>
void Write(std::vector<std::vector<T>> v, std::ostream *os) {
  for (const auto &i : v) {
    Write(i, os);
  }
}

class Package {
 public:
  Package() {}
  virtual ~Package() {}
  virtual void Pack(std::ostream *os) = 0;
};

class TensorInfo : public Package {
 public:
  explicit TensorInfo(const std::string &name, const std::vector<int> &shape,
                      const std::string &dtype, int lod_level)
      : name_(name), shape_(shape), dtype_(dtype), lod_level_(lod_level) {}
  void Pack(std::ostream *os) override {
    Write(name_, os);
    Write(shape_, os);
    Write(dtype_, os);
    Write<int>(lod_level_, os);
  }

  const std::string &name() const { return name_; }
  const std::vector<int> &shape() const { return shape_; }
  const std::string &dtype() const { return dtype_; }

 private:
  std::string name_;
  std::vector<int> shape_;
  std::string dtype_;
  int lod_level_;
};

template <typename T>
class TensorData : public Package {
 public:
  TensorData(const std::vector<T> &data,
             const std::vector<std::vector<size_t>> &lod)
      : data_(data), lod_(lod) {}
  void Pack(std::ostream *os) override {
    if (!lod_.empty()) Write(lod_, os);
    Write(data_, os);
  }
  const std::vector<T> &data() const { return data_; }
  const std::vector<std::vector<size_t>> &lod() const { return lod_; }

 private:
  std::vector<T> data_;
  std::vector<std::vector<size_t>> lod_;
};

class ReaderTest : public ::testing::Test {
 public:
  void SetUp();
  void TearDown();

  void ResetReader(int batch_size);

 protected:
  std::unique_ptr<paddle::debug::Reader> reader_;
  std::vector<std::shared_ptr<Package>> data_;

  size_t num_info_{0};

  boost::filesystem::path file_;
  std::string filename_;
};

void ReaderTest::SetUp() {
  // std::shared_ptr<Package> x_info = std::make_shared<TensorInfo>()
  std::hash<std::thread::id> hasher;
  std::string filename = "test_reader_tmp_file.bin." +
                         std::to_string(hasher(std::this_thread::get_id()));
  std::ofstream os(filename, std::ios::out | std::ios::binary);
  file_ = boost::filesystem::current_path() / filename;

  data_.emplace_back(
      std::make_shared<TensorInfo>("x", std::vector<int>({-1, 3}), "int64", 0));
  ++num_info_;
  data_.emplace_back(std::make_shared<TensorInfo>(
      "y", std::vector<int>({-1, 10}), "float32", 1));
  ++num_info_;

  std::random_device rd;
  std::mt19937 engine(rd());

  auto x_dist = std::uniform_int_distribution<int64_t>(0, 100);
  std::vector<int64_t> x_data(3);
  std::for_each(x_data.begin(), x_data.end(),
                [&x_dist, &engine](int64_t &v) { v = x_dist(engine); });
  data_.emplace_back(std::make_shared<TensorData<int64_t>>(
      x_data, std::vector<std::vector<size_t>>()));

  auto y_dist = std::uniform_real_distribution<float>(0, 1000);
  std::vector<float> y_data(10);
  std::for_each(y_data.begin(), y_data.end(),
                [&y_dist, &engine](float &v) { v = y_dist(engine); });
  data_.emplace_back(std::make_shared<TensorData<float>>(
      y_data, std::vector<std::vector<size_t>>({{10}})));

  Write(num_info_, &os);
  for (const auto &data : data_) {
    data->Pack(&os);
  }
}

void ReaderTest::ResetReader(int batch_size) {
  auto reader = std::unique_ptr<paddle::debug::Reader>(
      new paddle::debug::Reader(file_.string()));
  reader_ = std::move(reader);
  reader_->SetBatchSize(batch_size);
}

TEST_F(ReaderTest, test_get_batch_data) {
  ResetReader(1);

  const auto &data = reader_->data();
  reader_->NextBatch();

  for (size_t i = 0; i < data.size(); i++) {
    const auto &tensor = data[i];
    auto info = std::dynamic_pointer_cast<TensorInfo>(data_[i]);
    ASSERT_EQ(tensor.name, info->name());
    ASSERT_THAT(tensor.shape, ::testing::ElementsAreArray(info->shape()));
    paddle::PaddleDType dtype;
    paddle::debug::GetPaddleDType(info->dtype(), &dtype);
    ASSERT_EQ(tensor.dtype, dtype);
  }

  for (size_t i = 0; i < data.size(); i++) {
    const auto &tensor = data[i];
    std::vector<std::vector<size_t>> lod;
    if (tensor.dtype == paddle::PaddleDType::INT64) {
      auto original =
          std::dynamic_pointer_cast<TensorData<int64_t>>(data_[i + num_info_]);
      auto *data = static_cast<int64_t *>(tensor.data.data());
      auto size = tensor.data.length() / sizeof(int64_t);
      ASSERT_THAT(original->data(), ::testing::ElementsAreArray(data, size));
      lod = original->lod();
    } else if (tensor.dtype == paddle::PaddleDType::FLOAT32) {
      auto original =
          std::dynamic_pointer_cast<TensorData<float>>(data_[i + num_info_]);
      auto *data = static_cast<float *>(tensor.data.data());
      auto size = tensor.data.length() / sizeof(float);
      for (int j = 0; j < size; j++) {
        EXPECT_FLOAT_EQ(data[j], original->data()[j]);
        // std::cout << "========[" << j << "], [" << data[j] << "], ["
        //          << original->data()[j] << "]\n";
      }
      std::cout << std::endl;

      lod = original->lod();
    }

    if (!lod.empty()) {
      for (auto &item : lod) {
        item.insert(item.begin(), 0);
        std::partial_sum(item.begin(), item.end(), item.begin(),
                         std::plus<size_t>());
      }

      ASSERT_THAT(tensor.lod, ::testing::ElementsAreArray(lod));
    }
  }
}

void ReaderTest::TearDown() {
  // remote temp file
  boost::filesystem::remove(file_);
}
