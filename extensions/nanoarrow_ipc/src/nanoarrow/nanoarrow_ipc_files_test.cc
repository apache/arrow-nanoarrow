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

#include <sstream>
#include <stdio.h>

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

  std::string path_;
  int expected_return_code_;
  std::string expected_error_message_;
};

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

  const TestFile& param = GetParam();
  std::stringstream path_builder;
  path_builder << testing_dir << "/" << param.path_;

  FILE* file_ptr = fopen(path_builder.str().c_str(), "rb");
  ASSERT_NE(file_ptr, nullptr);

  struct ArrowIpcInputStream input;
  nanoarrow::UniqueArrayStream stream;
  ASSERT_EQ(ArrowIpcInputStreamInitFile(&input, file_ptr, true), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(stream.get(), &input, nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema schema;
  struct ArrowError error;
  int result = stream->get_schema(stream.get(), schema.get());
  if (result != NANOARROW_OK) {
    ASSERT_EQ(result, param.expected_return_code_);
    const char* msg = stream->get_last_error(stream.get());
    ASSERT_NE(msg, nullptr);
    ASSERT_EQ(std::string(msg), param.expected_error_message_);
    return;
  }

  std::vector<nanoarrow::UniqueArray> arrays;
  while (true) {
    nanoarrow::UniqueArray array;

    result = stream->get_next(stream.get(), array.get());
    if (result != NANOARROW_OK) {
      ASSERT_EQ(result, param.expected_return_code_);
      const char* msg = stream->get_last_error(stream.get());
      ASSERT_NE(msg, nullptr);
      ASSERT_EQ(std::string(msg), param.expected_error_message_);
      return;
    }

    arrays.push_back(std::move(array));
  }
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, ArrowTestingPathParameterizedTestFixture,
    ::testing::Values(
        // Comment to keep the first line from wrapping
        TestFile("data/arrow-ipc-stream/integration/1.0.0-littleendian/"
                 "generated_custom_metadata.arrow_file")

        // Comment to keep last line from wrapping
        ));
