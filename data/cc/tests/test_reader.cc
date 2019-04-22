#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include "debug_reader.h"

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

template <typename T>
void Write(const T &v, std::ostream *os) {
  auto bv = boost::endian::native_to_big(v);
  os->write(reinterpret_cast<char *>(&bv), sizeof(bv));
}

template <>
void Write(const std::string &s, std::ostream *os) {
  int size = s.size();
  size = boost::endian::native_to_big(size);
  os->write(reinterpret_cast<char *>(&size), sizeof(size));
  os->write(&s.front(), size);
}

template <typename T>
void Write(const std::vector<T> &v, std::ostream *os) {
  size_t size = v.size();
  size = boost::endian::native_to_big(size);
  os->write(reinterpret_cast<char *>(&size), sizeof(size));
  for (const auto &i : v) {
    auto bi = boost::endian::native_to_big(i);
    os->write(reinterpret_cast<char *>(&bi), sizeof(bi));
  }
}

template <typename T>
void Write(const std::vector<std::vector<T>> &v, std::ostream *os) {
  size_t size = v.size();
  size = boost::endian::native_to_big(size);
  os->write(reinterpret_cast<char *>(&size), sizeof(size));
  for (const auto &i : v) {
    WriteVector(i, os);
  }
}

class Package {
 public:
  Package() {}
  virtual ~Package() = 0;
  virtual void Unpack(std::ostream *os) = 0;
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
    Write(lod_level_, os);
  }

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
      : data_(data), lod(lod_) {}
  void Pack(std::ostream *os) override {
    Write(data_, os);
    Write(lod_, os);
  }

 private:
  std::vector<T> data_;
  std::vector<std::vector<size_t>> lod_;
};

class ReaderTest : public ::testing::Test {
 public:
  void SetUp();
  void TearDown();

 private:
  std::stringstream ss_;
};

void ReaderTest::SetUp() {}
