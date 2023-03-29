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

#include <arrow/c/bridge.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/api.h>
#include <arrow/table.h>
#include <gtest/gtest.h>

#include "nanoarrow.hpp"
#include "nanoarrow_ipc.h"

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

    // If the file was supposed to fail the read but did not, fail here
    if (expected_return_code_ != NANOARROW_OK) {
      GTEST_FAIL() << MakeError(NANOARROW_OK, "");
    }

    // Read the same file with Arrow C++
    auto options = ipc::IpcReadOptions::Defaults();
    auto fs = fs::LocalFileSystem();
    auto maybe_input_stream = fs.OpenInputStream(path_builder.str());
    if (!maybe_input_stream.ok()) {
      GTEST_FAIL() << maybe_input_stream.status().message();
    }

    auto maybe_reader = ipc::RecordBatchStreamReader::Open(*maybe_input_stream);
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
        // Files in data/arrow-ipc-stream/integration/1.0.0-littleendian/
        // should read without error and the data should match Arrow C++'s read
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

        // Files with features that are not yet supported (Dictionary encoding)
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
            "Schema message field with DictionaryEncoding not supported"),

        // Fuzzed files in data/arrow-ipc-stream. As a first pass test, we just make
        // sure that they don't crash our reader and don't succeed.
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-file-fuzz-5298734406172672"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-file-fuzz-5502930036326400"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-arrow-ipc-file-fuzz-6065820480962560.fuzz"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-file-fuzz-6537416932982784"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-file-fuzz-6598997234548736"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-4895056843112448"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-4904988668854272"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5077390980284416"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5085285868371968"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5151909853528064"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5183543628791808"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5435281763467264"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5634103970103296"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5651311318269952"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5675895545397248"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5677954426994688"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5682204130934784"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5719752899297280"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5738372907925504"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5756862809243648"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5768423720353792"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-5837681544396800"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6232191822725120"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6234449985142784"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6245758969577472"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6254629906808832"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6296372407697408"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6311624808595456.fuzz"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6321355259904000"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6321882936901632"),
        TestFile::ErrorAny("data/arrow-ipc-stream/"
                           "clusterfuzz-testcase-arrow-ipc-stream-fuzz-6487596632637440"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-file-fuzz-6674891504484352"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4757582821064704"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4831362862022656.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4851743764250624"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4889687236018176"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4961281405222912"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-4964779626856448"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5067615893192704"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5089431154589696"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5113616637100032"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5144746570022912"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5157190818332672.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5159348220461056"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5183404614352896"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5185274653179904"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5191432679981056"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5235940308811776.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5281967462023168"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5435281763467264"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5455172100423680"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5639621460099072"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5661776796712960"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5666296880168960"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5675895545397248"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5678890496557056"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5682204130934784"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5685159454310400"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5685713856888832"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5701512139636736"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5712457209479168"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5718685113384960"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5729978629226496"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-5760415636389888"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6204424660975616.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6310318291288064.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6311775452790784"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6321355259904000"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6440533038989312"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/"
            "clusterfuzz-testcase-minimized-arrow-ipc-stream-fuzz-6589380504977408.fuzz"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-09f72ba2a52b80366ab676364abec850fc668168"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-1fb75de2edd2815ad7a653684c449d814f39290e"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-2354085db0125113f04f7bd23f54b85cca104713"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-3c3f1b74f347ec6c8b0905e7126b9074b9dc5564"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-5e88bae6ac5250714e8c8bc73b9d67b949fadbb4"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-607e9caa76863a97f2694a769a1ae2fb83c55e02"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-74aec871d14bb6b07c72ea8f0e8c9f72cbe6b73c"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-bd7e00178af2d236fdf041fcc1fb30975bf8fbca"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-cb8cedb6ff8a6f164210c497d91069812ef5d6f8"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-f37e71777ad0324b55b99224f2c7ffb0107bdfa2"),
        TestFile::ErrorAny(
            "data/arrow-ipc-stream/crash-fd237566879dc60fff4d956d5fe3533d74a367f3")

        // Comment to keep last line from wrapping
        ));
