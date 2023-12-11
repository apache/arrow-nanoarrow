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

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <gtest/gtest.h>

#include "nanoarrow.hpp"
#include "nanoarrow_ipc.h"

#include "flatcc/portable/pendian_detect.h"

using namespace arrow;

// Utility to test an IPC stream written a as a file (where path does not include a
// prefix that might be specific where a specific system has arrow-testing checked out).
// This helper also checks a read that is supposed to be valid against what Arrow C++
// would read.
class TestFile {
 public:
  TestFile(std::string path, int expected_return_code, std::string expected_error_message)
      : path_(path),
        expected_return_code_(expected_return_code),
        expected_error_message_(expected_error_message) {}

  TestFile(std::string path) : TestFile(path, NANOARROW_OK, "") {}

  TestFile() : TestFile("") {}

  std::string CheckJSONGzFile() {
    size_t dot_pos = path_.find('.');
    return path_.substr(0, dot_pos) + std::string(".json.gz");
  }

  static TestFile OK(std::string path) { return TestFile(path); }

  static TestFile Err(int code, std::string path, std::string message = "__any__") {
    return TestFile(path, code, message);
  }

  static TestFile ErrorAny(std::string path) {
    return Err(std::numeric_limits<int>::max(), path);
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

  // Read a whole file into an ArrowBuffer. This allows us to avoid requiring Arrow C++
  // with filesystem.
  static void ReadFileBuffer(const std::string& path, ArrowBuffer* content) {
    std::ifstream infile(path, std::ios::in | std::ios::binary);
    do {
      content->size_bytes += infile.gcount();
      ASSERT_EQ(ArrowBufferReserve(content, 8096), NANOARROW_OK);
    } while (
        infile.read(reinterpret_cast<char*>(content->data + content->size_bytes), 8096));
    content->size_bytes += infile.gcount();
  }

  static void BufferInputStream(ArrowBuffer* src, std::shared_ptr<io::InputStream>* dst) {
    // Read the same data with Arrow C++.
    auto content_copy_wrapped = Buffer::Wrap<uint8_t>(src->data, src->size_bytes);
    auto buffer_reader = std::make_shared<io::BufferReader>(content_copy_wrapped);

    // Support Arrow 9.0.0 for Fedora and Centos7 images
#if ARROW_VERSION_MAJOR >= 10
    auto maybe_input_stream =
        io::RandomAccessFile::GetStream(buffer_reader, 0, content_copy_wrapped->size());
    if (!maybe_input_stream.ok()) {
      GTEST_FAIL() << maybe_input_stream.status().message();
    }

    std::shared_ptr<io::InputStream> input_stream = maybe_input_stream.ValueUnsafe();
#else
    std::shared_ptr<io::InputStream> input_stream =
        io::RandomAccessFile::GetStream(buffer_reader, 0, content_copy_wrapped->size());
#endif

    *dst = input_stream;
  }

  void TestEqualsArrowCpp(std::string dir_prefix) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    // Read using nanoarrow_ipc
    nanoarrow::UniqueBuffer content;
    ReadFileBuffer(path_builder.str(), content.get());

    struct ArrowIpcInputStream input;
    nanoarrow::UniqueArrayStream stream;
    ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, content.get()), NANOARROW_OK);
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

    // If the file was supposed to fail the read but did not, fail here
    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_FAIL() << MakeError(NANOARROW_OK, "");
    }

    // Read the same file with Arrow C++. Use the in-memory version to avoid
    // requiring Arrow C++ with filesystem.
    nanoarrow::UniqueBuffer content_copy;
    ReadFileBuffer(path_builder.str(), content_copy.get());
    std::shared_ptr<io::InputStream> input_stream;
    BufferInputStream(content_copy.get(), &input_stream);

    auto maybe_reader = ipc::RecordBatchStreamReader::Open(input_stream);
    if (!maybe_reader.ok()) {
      GTEST_FAIL() << maybe_reader.status().message();
    }

    auto maybe_table_arrow = maybe_reader.ValueUnsafe()->ToTable();
    if (!maybe_table_arrow.ok()) {
      GTEST_FAIL() << maybe_table_arrow.status().message();
    }

    // Make a Table from the our vector of arrays
    auto maybe_schema = ImportSchema(schema.get());
    if (!maybe_schema.ok()) {
      GTEST_FAIL() << maybe_schema.status().message();
    }

    ASSERT_TRUE(maybe_table_arrow.ValueUnsafe()->schema()->Equals(**maybe_schema, true));

    std::vector<std::shared_ptr<RecordBatch>> batches;
    for (auto& array : arrays) {
      auto maybe_batch = ImportRecordBatch(array.get(), *maybe_schema);
      batches.push_back(std::move(*maybe_batch));
    }

    auto maybe_table = Table::FromRecordBatches(*maybe_schema, batches);
    EXPECT_TRUE(maybe_table.ValueUnsafe()->Equals(**maybe_table_arrow, true));
  }

  void TestCheckJSON(std::string dir_prefix) {
    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_SKIP() << path_ << " is not currently supported by the IPC reader";
    }

    // Read IPC file into a buffer
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;
    nanoarrow::UniqueBuffer content;
    ReadFileBuffer(path_builder.str(), content.get());

    // Read .json.gz file into a buffer
    path_builder.str("");
    path_builder << dir_prefix << "/" << CheckJSONGzFile();
    nanoarrow::UniqueBuffer json_gz_content;
    ReadFileBuffer(path_builder.str(), json_gz_content.get());

    // Use Arrow C++ to decompress with gzip
    std::shared_ptr<io::InputStream> gz_input_stream;
    BufferInputStream(json_gz_content.get(), &gz_input_stream);

    auto maybe_gzip = arrow::util::Codec::Create(arrow::Compression::GZIP);
    if (!maybe_gzip.ok()) {
      GTEST_FAIL() << maybe_gzip.status().message();
    }

    auto maybe_input =
        io::CompressedInputStream::Make(maybe_gzip->get(), gz_input_stream);
    if (!maybe_input.ok()) {
      GTEST_FAIL() << maybe_input.status().message();
    }

    std::stringstream testing_json;
    auto input = *maybe_input;
    char buf[8096];
    int64_t bytes_read = 0;
    do {
      auto maybe_bytes_read = input->Read(sizeof(buf), buf);
      if (!maybe_bytes_read.ok()) {
        GTEST_FAIL() << maybe_bytes_read.status().message();
      }

      bytes_read = *maybe_bytes_read;
      testing_json << std::string(buf, bytes_read);
    } while (bytes_read > 0);
  }

  bool Check(int result, std::string msg) {
    return (expected_return_code_ == std::numeric_limits<int>::max() &&
            result != NANOARROW_OK) ||
           (result == expected_return_code_ && msg == expected_error_message_) ||
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

void InitArrowTestingPath(std::ostream& builder) {
  const char* testing_dir = getenv("NANOARROW_ARROW_TESTING_DIR");
  if (testing_dir == nullptr || strlen(testing_dir) == 0) {
    GTEST_SKIP() << "NANOARROW_ARROW_TESTING_DIR environment variable not set";
  }

  builder << testing_dir;
}

class TestFileFixture : public ::testing::TestWithParam<TestFile> {
 protected:
  TestFile test_file;
};

TEST_P(TestFileFixture, NanoarrowIpcTestFileNativeEndian) {
  std::stringstream dir_builder;
  InitArrowTestingPath(dir_builder);

#if defined(__BIG_ENDIAN__)
  dir_builder << "/data/arrow-ipc-stream/integration/1.0.0-bigendian";
#else
  dir_builder << "/data/arrow-ipc-stream/integration/1.0.0-littleendian";
#endif
  TestFile param = GetParam();
  param.TestEqualsArrowCpp(dir_builder.str());
}

TEST_P(TestFileFixture, NanoarrowIpcTestFileSwapEndian) {
  std::stringstream dir_builder;
  InitArrowTestingPath(dir_builder);

#if defined(__BIG_ENDIAN__)
  dir_builder << "/data/arrow-ipc-stream/integration/1.0.0-littleendian";
#else
  dir_builder << "/data/arrow-ipc-stream/integration/1.0.0-bigendian";
#endif
  TestFile param = GetParam();
  param.TestEqualsArrowCpp(dir_builder.str());
}

TEST_P(TestFileFixture, NanoarrowIpcTestFileCheckJSON) {
  std::stringstream dir_builder;
  InitArrowTestingPath(dir_builder);
  TestFile file = GetParam();

  if (file.expected_return_code_ != NANOARROW_OK) {
    GTEST_SKIP() << file.path_ << " is not currently supported by the IPC reader";
  }
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, TestFileFixture,
    ::testing::Values(
        // Files in data/arrow-ipc-stream/integration/1.0.0-(little|big)endian/
        // should read without error and the data should match Arrow C++'s read
        TestFile::OK("generated_custom_metadata.stream"),
        TestFile::OK("generated_datetime.stream"),
        TestFile::OK("generated_decimal.stream"),
        TestFile::OK("generated_decimal256.stream"),

        TestFile::OK("generated_duplicate_fieldnames.stream"),
        TestFile::OK("generated_interval.stream"),
        TestFile::OK("generated_map_non_canonical.stream"),
        TestFile::OK("generated_map.stream"),
        TestFile::OK("generated_nested_large_offsets.stream"),
        TestFile::OK("generated_nested.stream"),
        TestFile::OK("generated_null_trivial.stream"),
        TestFile::OK("generated_null.stream"),
        TestFile::OK("generated_primitive_large_offsets.stream"),
        TestFile::OK("generated_primitive_no_batches.stream"),
        TestFile::OK("generated_primitive_zerolength.stream"),
        TestFile::OK("generated_primitive.stream"),
        TestFile::OK("generated_recursive_nested.stream"),
        TestFile::OK("generated_union.stream"),

        // Files with features that are not yet supported (Dictionary encoding)
        TestFile::NotSupported(
            "generated_dictionary_unsigned.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "generated_dictionary.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "generated_nested_dictionary.stream",
            "Schema message field with DictionaryEncoding not supported"),
        TestFile::NotSupported(
            "generated_extension.stream",
            "Schema message field with DictionaryEncoding not supported")
        // Comment to keep last line from wrapping
        ));
