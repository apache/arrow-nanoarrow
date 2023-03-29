// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <fstream>
#include <sstream>

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>

#include "nanoarrow.hpp"
#include "nanoarrow_ipc.h"

using namespace arrow;

class TestFile {
 public:
  TestFile(std::string path, int expected_return_code, std::string expected_error_message)
      : path_(path),
        expected_return_code_(expected_return_code),
        expected_error_message_(expected_error_message) {}

  TestFile(std::string path) : TestFile(path, NANOARROW_OK, "") {}

  TestFile() : TestFile("") {}

  static TestFile OK(std::string path) { return TestFile(path); }

  static TestFile Err(int code, std::string path, std::string message = "__any__") {
    return TestFile(path, code, message);
  }

  static TestFile Invalid(std::string path, std::string message = "__any__") {
    return Err(EINVAL, path, message);
  }

  static TestFile NotSupported(std::string path, std::string message = "__any__") {
    return Err(ENOTSUP, path, message);
  }

  static TestFile NoData(std::string path, std::string message = "__any__") {
    return Err(ENODATA, path, message);
  }

  void Test(std::string dir_prefix) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    // Read the whole file into an ArrowBuffer
    std::ifstream infile(path_builder.str(), std::ios::in | std::ios::binary);
    nanoarrow::UniqueBuffer buf;
    do {
      buf->size_bytes += infile.gcount();
      ArrowBufferReserve(buf.get(), 8096);
    } while (infile.read(reinterpret_cast<char*>(buf->data + buf->size_bytes), 8096));
    buf->size_bytes += infile.gcount();

    struct ArrowIpcInputStream input;
    nanoarrow::UniqueArrayStream stream;
    ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, buf.get()), NANOARROW_OK);
    ASSERT_EQ(ArrowIpcArrayStreamReaderInit(stream.get(), &input, nullptr), NANOARROW_OK);

    nanoarrow::UniqueSchema schema;
    int result = stream->get_schema(stream.get(), schema.get());
    if (result != NANOARROW_OK) {
      std::string err(stream->get_last_error(stream.get()));
      if (Check(result, err)) {
        return;
      }

      GTEST_FAIL() << MakeError(result, err);
    }

    std::vector<nanoarrow::UniqueArray> arrays;
    while (true) {
      nanoarrow::UniqueArray array;

      result = stream->get_next(stream.get(), array.get());
      if (result != NANOARROW_OK) {
        std::string err(stream->get_last_error(stream.get()));
        if (Check(result, err)) {
          return;
        }

        GTEST_FAIL() << MakeError(result, err);
      }

      if (array->release == nullptr) {
        break;
      }

      arrays.push_back(std::move(array));
    }

    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_FAIL() << MakeError(NANOARROW_OK, "");
    }
  }

  bool Check(int result, std::string msg) {
    return (result == expected_return_code_ && msg == expected_error_message_) ||
           (result == expected_return_code_ && expected_error_message_ == "__any__");
  }

  std::string MakeError(int result, std::string msg) {
    std::stringstream err;
    err << "Expected file '" << path_ << "' to return code " << expected_return_code_
        << " and error message '" << expected_error_message_ << "' but got return code "
        << result << " and error message '" << msg << "'";
    return err.str();
  }

  std::string path_;
  int expected_return_code_;
  std::string expected_error_message_;
};

std::ostream& operator<<(std::ostream& os, const TestFile& obj) {
  os << obj.path_;
  return os;
}

class ArrowTestingPathParameterizedTestFixture
    : public ::testing::TestWithParam<TestFile> {
 protected:
  TestFile test_file;
};

TEST_P(ArrowTestingPathParameterizedTestFixture, NanoarrowIpcTestFile) {
  const char* testing_dir = getenv("NANOARROW_ARROW_TESTING_DIR");
  if (testing_dir == nullptr || strlen(testing_dir) == 0) {
    GTEST_SKIP() << "NANOARROW_ARROW_TESTING_DIR environment variable not set";
  }

  TestFile param = GetParam();
  param.Test(testing_dir);
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, ArrowTestingPathParameterizedTestFixture,
    ::testing::Values(
        // Comment to keep the first line from wrapping
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_custom_metadata.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_datetime.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_decimal.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_decimal256.stream"),

        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_duplicate_fieldnames.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_interval.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_map_non_canonical.stream"),
        TestFile::OK(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/generated_map.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_nested_large_offsets.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_nested.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_null_trivial.stream"),
        TestFile::OK(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/generated_null.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_primitive_large_offsets.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_primitive_no_batches.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_primitive_zerolength.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_primitive.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_recursive_nested.stream"),
        TestFile::OK("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                     "generated_union.stream"),

        // Dictionary encoding not yet supported
        TestFile::NotSupported(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/"
            "generated_dictionary_unsigned.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/"
            "generated_dictionary.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/"
            "generated_nested_dictionary.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "data/arrow-ipc-stream/integration/1.0.0-littleendian/"
            "generated_extension.stream",
            "Schema message field with DictionaryEncoding not supported")

        // Comment to keep last line from wrapping
        ));
