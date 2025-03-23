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

#include <zlib.h>

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_ipc.hpp"
#include "nanoarrow/nanoarrow_testing.hpp"

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
    size_t dot_pos = path_.rfind('.');
    return path_.substr(0, dot_pos) + std::string(".json.gz");
  }

  ArrowErrorCode GetArrowArrayStreamIPC(struct ArrowBuffer* content,
                                        ArrowArrayStream* out, ArrowError* error) {
    nanoarrow::ipc::UniqueInputStream input;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcInputStreamInitBuffer(input.get(), content), error);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcArrayStreamReaderInit(out, input.get(), nullptr), error);
    return NANOARROW_OK;
  }

  ArrowErrorCode GetArrowArrayStreamIPC(const std::string& dir_prefix,
                                        ArrowArrayStream* out, ArrowError* error) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    // Read using nanoarrow_ipc
    nanoarrow::UniqueBuffer content;
    NANOARROW_RETURN_NOT_OK(ReadFileBuffer(path_builder.str(), content.get(), error));
    return GetArrowArrayStreamIPC(content.get(), out, error);
  }

  ArrowErrorCode ReadArrowArrayStreamIPC(struct ArrowArrayStream* stream,
                                         struct ArrowSchema* schema,
                                         std::vector<nanoarrow::UniqueArray>* arrays,
                                         ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetSchema(stream, schema, error));

    while (true) {
      nanoarrow::UniqueArray array;

      NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(stream, array.get(), error));

      if (array->release == nullptr) {
        break;
      }

      arrays->push_back(std::move(array));
    }
    return NANOARROW_OK;
  }

  ArrowErrorCode ReadArrowArrayStreamIPC(const std::string& dir_prefix,
                                         struct ArrowSchema* schema,
                                         std::vector<nanoarrow::UniqueArray>* arrays,
                                         ArrowError* error) {
    nanoarrow::UniqueArrayStream stream;
    NANOARROW_RETURN_NOT_OK(GetArrowArrayStreamIPC(dir_prefix, stream.get(), error));
    return ReadArrowArrayStreamIPC(stream.get(), schema, arrays, error);
  }

  ArrowErrorCode GetArrowArrayStreamCheckJSON(const std::string& dir_prefix,
                                              ArrowArrayStream* out, ArrowError* error) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << CheckJSONGzFile();

    // Read .json.gz file into a buffer
    nanoarrow::UniqueBuffer json_content;
    NANOARROW_RETURN_NOT_OK(
        ReadGzFileBuffer(path_builder.str(), json_content.get(), error));

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

  static ArrowErrorCode ReadGzFileBuffer(const std::string& path, ArrowBuffer* dst,
                                         ArrowError* error) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (file == NULL) {
      ArrowErrorSet(error, "Failed to open '%s'", path.c_str());
      return EINVAL;
    }

    char buf[8096];
    int out_len = 0;
    do {
      out_len = gzread(file, buf, sizeof(buf));
      if (out_len < 0) {
        gzclose(file);
        ArrowErrorSet(error, "gzread() returned %d", out_len);
        return EIO;
      }

      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppend(dst, buf, out_len), error);
    } while (out_len > 0);

    gzclose(file);
    return NANOARROW_OK;
  }

  // Create an arrow::io::InputStream wrapper around an ArrowBuffer
  static std::shared_ptr<io::InputStream> BufferInputStream(ArrowBuffer* src) {
    auto content_copy_wrapped = Buffer::Wrap<uint8_t>(src->data, src->size_bytes);
    return std::make_shared<io::BufferReader>(content_copy_wrapped);
  }

  ArrowErrorCode WriteNanoarrowStream(const nanoarrow::UniqueSchema& schema,
                                      const std::vector<nanoarrow::UniqueArray>& arrays,
                                      struct ArrowBuffer* buffer,
                                      struct ArrowError* error) {
    nanoarrow::ipc::UniqueOutputStream output_stream;
    NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamInitBuffer(output_stream.get(), buffer));

    nanoarrow::ipc::UniqueWriter writer;
    NANOARROW_RETURN_NOT_OK(ArrowIpcWriterInit(writer.get(), output_stream.get()));

    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), error));

    NANOARROW_RETURN_NOT_OK(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), error));
    for (const auto& array : arrays) {
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayViewSetArray(array_view.get(), array.get(), error));

      NANOARROW_RETURN_NOT_OK(
          ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), error));
    }
    return ArrowIpcWriterWriteArrayView(writer.get(), nullptr, error);
  }

  void TestEqualsArrowCpp(const std::string& dir_prefix) {
    std::stringstream path_builder;
    path_builder << dir_prefix << "/" << path_;

    ArrowError error;
    ArrowErrorInit(&error);

    // Read using nanoarrow_ipc
    nanoarrow::UniqueSchema schema;
    std::vector<nanoarrow::UniqueArray> arrays;
    int result = ReadArrowArrayStreamIPC(dir_prefix, schema.get(), &arrays, &error);
    if (result != NANOARROW_OK) {
      if (Check(result, error.message)) {
        return;
      }
      GTEST_FAIL() << MakeError(result, error.message);
    }

    // If the file was supposed to fail the read but did not, fail here
    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_FAIL() << MakeError(NANOARROW_OK, "");
    }

    // Write back to a buffer using nanoarrow
    nanoarrow::UniqueBuffer roundtripped;
    ASSERT_EQ(WriteNanoarrowStream(schema, arrays, roundtripped.get(), &error),
              NANOARROW_OK)
        << error.message;

    // Read the same file with Arrow C++
    auto maybe_table_arrow = ReadTable(io::ReadableFile::Open(path_builder.str()));
    {
      SCOPED_TRACE("Read the same file with Arrow C++");
      FAIL_RESULT_NOT_OK(maybe_table_arrow);
      AssertEqualsTable(std::move(schema), std::move(arrays),
                        maybe_table_arrow.ValueUnsafe());
    }

    auto maybe_table_roundtripped = ReadTable(BufferInputStream(roundtripped.get()));
    {
      SCOPED_TRACE("Read the roundtripped buffer using Arrow C++");
      FAIL_RESULT_NOT_OK(maybe_table_roundtripped);

      AssertEqualsTable(maybe_table_roundtripped.ValueUnsafe(),
                        maybe_table_arrow.ValueUnsafe());
    }

    nanoarrow::UniqueSchema roundtripped_schema;
    std::vector<nanoarrow::UniqueArray> roundtripped_arrays;
    {
      SCOPED_TRACE("Read the roundtripped buffer using nanoarrow");
      nanoarrow::UniqueArrayStream array_stream;
      ASSERT_EQ(GetArrowArrayStreamIPC(roundtripped.get(), array_stream.get(), &error),
                NANOARROW_OK);
      ASSERT_EQ(ReadArrowArrayStreamIPC(array_stream.get(), roundtripped_schema.get(),
                                        &roundtripped_arrays, &error),
                NANOARROW_OK);

      AssertEqualsTable(std::move(roundtripped_schema), std::move(roundtripped_arrays),
                        maybe_table_arrow.ValueUnsafe());
    }
  }

  Result<std::shared_ptr<Table>> ReadTable(
      Result<std::shared_ptr<io::InputStream>> maybe_input_stream) {
    ARROW_ASSIGN_OR_RAISE(auto input_stream, maybe_input_stream);
    ARROW_ASSIGN_OR_RAISE(auto reader, ipc::RecordBatchStreamReader::Open(input_stream));
    return reader->ToTable();
  }

  Result<std::shared_ptr<Table>> ToTable(nanoarrow::UniqueSchema schema,
                                         std::vector<nanoarrow::UniqueArray> arrays) {
    ARROW_ASSIGN_OR_RAISE(auto arrow_schema, ImportSchema(schema.get()));

    std::vector<std::shared_ptr<RecordBatch>> batches(arrays.size());
    size_t i = 0;
    for (auto& array : arrays) {
      ARROW_ASSIGN_OR_RAISE(auto batch, ImportRecordBatch(array.get(), arrow_schema));
      batches[i++] = std::move(batch);
    }
    return Table::FromRecordBatches(std::move(arrow_schema), std::move(batches));
  }

  void AssertEqualsTable(const std::shared_ptr<Table>& actual,
                         const std::shared_ptr<Table>& expected) {
    ASSERT_TRUE(actual->schema()->Equals(expected->schema(), true));
    EXPECT_TRUE(actual->Equals(*expected, true));
  }

  void AssertEqualsTable(nanoarrow::UniqueSchema schema,
                         std::vector<nanoarrow::UniqueArray> arrays,
                         const std::shared_ptr<Table>& expected) {
    auto maybe_table = ToTable(std::move(schema), std::move(arrays));
    FAIL_RESULT_NOT_OK(maybe_table);
    AssertEqualsTable(maybe_table.ValueUnsafe(), expected);
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

class TestEndianFileFixture : public ::testing::TestWithParam<TestFile> {
 protected:
  TestFile test_file;
};

TEST_P(TestEndianFileFixture, NanoarrowIpcTestFileNativeEndian) {
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

TEST_P(TestEndianFileFixture, NanoarrowIpcTestFileSwapEndian) {
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

TEST_P(TestEndianFileFixture, NanoarrowIpcTestFileCheckJSON) {
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
    NanoarrowIpcTest, TestEndianFileFixture,
    ::testing::Values(
        // Files in data/arrow-ipc-stream/integration/1.0.0-(little|big)endian/
        // should read without error and the data should match Arrow C++'s read.
        // Also write the stream to a buffer and check Arrow C++'s read of that.
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

// Files not related to endianness (i.e., only need testing once)
class TestFileFixture : public ::testing::TestWithParam<TestFile> {
 protected:
  TestFile test_file;
};

TEST_P(TestFileFixture, NanoarrowIpcTestFileEqualsArrowCpp) {
  std::stringstream dir_builder;
  ArrowError error;
  ArrowErrorInit(&error);
  if (InitArrowTestingPath(dir_builder, &error) != NANOARROW_OK) {
    GTEST_SKIP() << error.message;
  }

  dir_builder << "/data/arrow-ipc-stream/integration/";
  TestFile param = GetParam();
  param.TestEqualsArrowCpp(dir_builder.str());
}

TEST_P(TestFileFixture, NanoarrowIpcTestFileIPCCheckJSON) {
  std::stringstream dir_builder;
  ArrowError error;
  ArrowErrorInit(&error);
  if (InitArrowTestingPath(dir_builder, &error) != NANOARROW_OK) {
    GTEST_SKIP() << error.message;
  }

  dir_builder << "/data/arrow-ipc-stream/integration/";
  TestFile param = GetParam();
  param.TestIPCCheckJSON(dir_builder.str());
}

// At least one Windows MSVC version does not allow the #if defined()
// to be within a macro invocation, so we define these two cases
// with some repetition.
#if defined(NANOARROW_IPC_WITH_ZSTD)
INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, TestFileFixture,
    ::testing::Values(
        // Testing of other files
        TestFile::OK("2.0.0-compression/generated_uncompressible_zstd.stream"),
        TestFile::OK("2.0.0-compression/generated_zstd.stream"),
        TestFile::OK("0.17.1/generated_union.stream"),
        TestFile::OK("0.14.1/generated_datetime.stream"),
        TestFile::OK("0.14.1/generated_decimal.stream"),
        TestFile::OK("0.14.1/generated_interval.stream"),
        TestFile::OK("0.14.1/generated_map.stream"),
        TestFile::OK("0.14.1/generated_nested.stream"),
        TestFile::OK("0.14.1/generated_primitive.stream"),
        TestFile::OK("0.14.1/generated_primitive_no_batches.stream"),
        TestFile::OK("0.14.1/generated_primitive_zerolength.stream")
        // Comment to keep line from wrapping
        ));
#else
INSTANTIATE_TEST_SUITE_P(NanoarrowIpcTest, TestFileFixture,
                         ::testing::Values(
                             // Testing of other files
                             TestFile::OK("0.17.1/generated_union.stream"),
                             TestFile::OK("0.14.1/generated_datetime.stream"),
                             TestFile::OK("0.14.1/generated_decimal.stream"),
                             TestFile::OK("0.14.1/generated_interval.stream"),
                             TestFile::OK("0.14.1/generated_map.stream"),
                             TestFile::OK("0.14.1/generated_nested.stream"),
                             TestFile::OK("0.14.1/generated_primitive.stream"),
                             TestFile::OK("0.14.1/generated_primitive_no_batches.stream"),
                             TestFile::OK("0.14.1/generated_primitive_zerolength.stream")
                             // Comment to keep line from wrapping
                             ));
#endif

#endif
