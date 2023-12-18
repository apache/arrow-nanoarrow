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
#include "nanoarrow_testing.hpp"

#include "flatcc/portable/pendian_detect.h"

using namespace arrow;

// Helpers for reporting Arrow C++ Result failures
#define FAIL_RESULT_NOT_OK(expr) \
  if (!(expr).ok()) GTEST_FAIL() << (expr).status().message()

#define NANOARROW_RETURN_ARROW_RESULT_NOT_OK(expr, error_expr)            \
  if (!(expr).ok()) {                                                     \
    ArrowErrorSet((error_expr), "%s", (expr).status().message().c_str()); \
    return EINVAL;                                                        \
  }

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

  std::string CheckJSONGzFile() {
    size_t dot_pos = path_.find('.');
    return path_.substr(0, dot_pos) + std::string(".json.gz");
  }

  ArrowErrorCode GetArrowArrayStreamIPC(const std::string& dir_prefix,
                                        ArrowArrayStream* out, ArrowError* error) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    // Read using nanoarrow_ipc
    nanoarrow::UniqueBuffer content;
    NANOARROW_RETURN_NOT_OK(ReadFileBuffer(path_builder.str(), content.get(), error));

    struct ArrowIpcInputStream input;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcInputStreamInitBuffer(&input, content.get()), error);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcArrayStreamReaderInit(out, &input, nullptr), error);
    return NANOARROW_OK;
  }

  ArrowErrorCode GetArrowArrayStreamCheckJSON(const std::string& dir_prefix,
                                              ArrowArrayStream* out, ArrowError* error) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << CheckJSONGzFile();

    // Read .json.gz file into a buffer
    nanoarrow::UniqueBuffer json_gz_content;
    NANOARROW_RETURN_NOT_OK(
        ReadFileBuffer(path_builder.str(), json_gz_content.get(), error));

    // Decompress into a JSON string
    nanoarrow::UniqueBuffer json_content;
    NANOARROW_RETURN_NOT_OK(UnGZIP(json_gz_content.get(), json_content.get(), error));

    std::string json_string(reinterpret_cast<char*>(json_content->data),
                            json_content->size_bytes);

    // Use testing util to populate the array stream
    nanoarrow::testing::TestingJSONReader reader;
    NANOARROW_RETURN_NOT_OK(reader.ReadDataFile(
        json_string, out, nanoarrow::testing::TestingJSONReader::kNumBatchReadAll,
        error));
    return NANOARROW_OK;
  }

  // Read a whole file into an ArrowBuffer
  static ArrowErrorCode ReadFileBuffer(const std::string& path, ArrowBuffer* content,
                                       ArrowError* error) {
    std::ifstream infile(path, std::ios::in | std::ios::binary);
    do {
      content->size_bytes += infile.gcount();
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(content, 8096), error);
    } while (
        infile.read(reinterpret_cast<char*>(content->data + content->size_bytes), 8096));
    content->size_bytes += infile.gcount();

    return NANOARROW_OK;
  }

  // Create an arrow::io::InputStream wrapper around an ArrowBuffer
  static std::shared_ptr<io::InputStream> BufferInputStream(ArrowBuffer* src) {
    auto content_copy_wrapped = Buffer::Wrap<uint8_t>(src->data, src->size_bytes);
    return std::make_shared<io::BufferReader>(content_copy_wrapped);
  }

  // Decompress gzipped buffer content (currently uses Arrow C++)
  static ArrowErrorCode UnGZIP(ArrowBuffer* src, ArrowBuffer* dst, ArrowError* error) {
    auto maybe_gzip = arrow::util::Codec::Create(arrow::Compression::GZIP);
    NANOARROW_RETURN_ARROW_RESULT_NOT_OK(maybe_gzip, error);

    std::shared_ptr<io::InputStream> gz_input_stream = BufferInputStream(src);

    auto maybe_input =
        io::CompressedInputStream::Make(maybe_gzip->get(), gz_input_stream);
    NANOARROW_RETURN_ARROW_RESULT_NOT_OK(maybe_input, error);

    std::stringstream testing_json;
    auto input = *maybe_input;
    int64_t bytes_read = 0;
    do {
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(dst, 8096), error);

      auto maybe_bytes_read = input->Read(8096, dst->data + dst->size_bytes);
      NANOARROW_RETURN_ARROW_RESULT_NOT_OK(maybe_bytes_read, error);

      bytes_read = *maybe_bytes_read;
      dst->size_bytes += bytes_read;
    } while (bytes_read > 0);

    return NANOARROW_OK;
  }

  void TestEqualsArrowCpp(const std::string& dir_prefix) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    ArrowError error;
    ArrowErrorInit(&error);

    // Read using nanoarrow_ipc
    nanoarrow::UniqueArrayStream stream;
    ASSERT_EQ(GetArrowArrayStreamIPC(dir_prefix, stream.get(), &error), NANOARROW_OK)
        << error.message;

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

    // Read the same file with Arrow C++
    nanoarrow::UniqueBuffer content_copy;
    ASSERT_EQ(ReadFileBuffer(path_builder.str(), content_copy.get(), &error),
              NANOARROW_OK)
        << error.message;
    std::shared_ptr<io::InputStream> input_stream = BufferInputStream(content_copy.get());

    auto maybe_reader = ipc::RecordBatchStreamReader::Open(input_stream);
    FAIL_RESULT_NOT_OK(maybe_reader);

    auto maybe_table_arrow = maybe_reader.ValueUnsafe()->ToTable();
    FAIL_RESULT_NOT_OK(maybe_table_arrow);

    // Make a Table from the our vector of arrays
    auto maybe_schema = ImportSchema(schema.get());
    FAIL_RESULT_NOT_OK(maybe_schema);

    ASSERT_TRUE(maybe_table_arrow.ValueUnsafe()->schema()->Equals(**maybe_schema, true));

    std::vector<std::shared_ptr<RecordBatch>> batches;
    for (auto& array : arrays) {
      auto maybe_batch = ImportRecordBatch(array.get(), *maybe_schema);
      FAIL_RESULT_NOT_OK(maybe_batch);

      batches.push_back(std::move(*maybe_batch));
    }

    auto maybe_table = Table::FromRecordBatches(*maybe_schema, batches);
    FAIL_RESULT_NOT_OK(maybe_table);

    EXPECT_TRUE(maybe_table.ValueUnsafe()->Equals(**maybe_table_arrow, true));
  }

  void TestIPCCheckJSON(const std::string& dir_prefix) {
    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_SKIP() << path_ << " is not currently supported by the IPC reader";
    }

    ArrowError error;
    ArrowErrorInit(&error);

    nanoarrow::UniqueArrayStream ipc_stream;
    ASSERT_EQ(GetArrowArrayStreamIPC(dir_prefix, ipc_stream.get(), &error), NANOARROW_OK)
        << error.message;

    nanoarrow::UniqueArrayStream json_stream;
    int result = GetArrowArrayStreamCheckJSON(dir_prefix, json_stream.get(), &error);
    // Skip instead of fail for ENOTSUP
    if (result == ENOTSUP) {
      GTEST_SKIP() << "File contains type(s) not supported by the testing JSON reader: "
                   << error.message;
    }
    ASSERT_EQ(result, NANOARROW_OK) << error.message;

    // Use testing utility to compare
    nanoarrow::testing::TestingJSONComparison comparison;
    ASSERT_EQ(comparison.CompareArrayStream(ipc_stream.get(), json_stream.get(), &error),
              NANOARROW_OK)
        << error.message;

    std::stringstream differences;
    comparison.WriteDifferences(differences);
    EXPECT_EQ(comparison.num_differences(), 0)
        << "CompareArrayStream() found " << comparison.num_differences()
        << " difference(s):\n"
        << differences.str();
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

// For better testing output
std::ostream& operator<<(std::ostream& os, const TestFile& obj) {
  os << obj.path_;
  return os;
}

// Start building the arrow-testing path or error if the environment
// variable was not set
ArrowErrorCode InitArrowTestingPath(std::ostream& builder, ArrowError* error) {
  const char* testing_dir = getenv("NANOARROW_ARROW_TESTING_DIR");
  if (testing_dir == nullptr || strlen(testing_dir) == 0) {
    ArrowErrorSet(error, "NANOARROW_ARROW_TESTING_DIR environment variable not set");
    return ENOENT;
  }

  builder << testing_dir;
  return NANOARROW_OK;
}

class TestFileFixture : public ::testing::TestWithParam<TestFile> {
 protected:
  TestFile test_file;
};

TEST_P(TestFileFixture, NanoarrowIpcTestFileNativeEndian) {
  std::stringstream dir_builder;
  ArrowError error;
  ArrowErrorInit(&error);
  if (InitArrowTestingPath(dir_builder, &error) != NANOARROW_OK) {
    GTEST_SKIP() << error.message;
  }

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
  ArrowError error;
  ArrowErrorInit(&error);
  if (InitArrowTestingPath(dir_builder, &error) != NANOARROW_OK) {
    GTEST_SKIP() << error.message;
  }

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
  ArrowError error;
  ArrowErrorInit(&error);
  if (InitArrowTestingPath(dir_builder, &error) != NANOARROW_OK) {
    GTEST_SKIP() << error.message;
  }

  dir_builder << "/data/arrow-ipc-stream/integration/1.0.0-littleendian";

  TestFile param = GetParam();
  param.TestIPCCheckJSON(dir_builder.str());
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
