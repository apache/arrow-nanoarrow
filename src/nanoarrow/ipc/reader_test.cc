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

#include <gtest/gtest.h>

#include <stdio.h>

#include <string>

#include "nanoarrow/nanoarrow_ipc.hpp"

static uint8_t kSimpleSchema[] = {
    0xff, 0xff, 0xff, 0xff, 0x10, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x0e, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x0c, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x84, 0xff,
    0xff, 0xff, 0x18, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x00, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x6b, 0x65, 0x79, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x00, 0x18, 0x00,
    0x08, 0x00, 0x06, 0x00, 0x07, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x14, 0x00,
    0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x14, 0x00, 0x00, 0x00, 0x70, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x63, 0x6f, 0x6c, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x76, 0x61, 0x6c,
    0x75, 0x65, 0x5f, 0x66, 0x69, 0x65, 0x6c, 0x64, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
    0x00, 0x00, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x6b, 0x65, 0x79, 0x5f, 0x66, 0x69, 0x65,
    0x6c, 0x64, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x07, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

static uint8_t kSimpleRecordBatch[] = {
    0xff, 0xff, 0xff, 0xff, 0x88, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x16, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x18, 0x00, 0x0c, 0x00,
    0x04, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

alignas(8) static uint8_t kDictionarySchema[] = {
    0xff, 0xff, 0xff, 0xff, 0x50, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x0e, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x0c, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0xb0, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00,
    0x0c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x8c, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x7e, 0x00, 0x00, 0x00, 0x41, 0x0a, 0x33, 0x0a, 0x32, 0x36,
    0x33, 0x31, 0x37, 0x30, 0x0a, 0x31, 0x39, 0x37, 0x38, 0x38, 0x38, 0x0a, 0x35, 0x0a,
    0x55, 0x54, 0x46, 0x2d, 0x38, 0x0a, 0x35, 0x33, 0x31, 0x0a, 0x31, 0x0a, 0x35, 0x33,
    0x31, 0x0a, 0x31, 0x0a, 0x32, 0x35, 0x34, 0x0a, 0x31, 0x30, 0x32, 0x36, 0x0a, 0x31,
    0x0a, 0x32, 0x36, 0x32, 0x31, 0x35, 0x33, 0x0a, 0x35, 0x0a, 0x6e, 0x61, 0x6d, 0x65,
    0x73, 0x0a, 0x31, 0x36, 0x0a, 0x31, 0x0a, 0x32, 0x36, 0x32, 0x31, 0x35, 0x33, 0x0a,
    0x38, 0x0a, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x63, 0x6f, 0x6c, 0x0a, 0x32, 0x35, 0x34,
    0x0a, 0x31, 0x30, 0x32, 0x36, 0x0a, 0x35, 0x31, 0x31, 0x0a, 0x31, 0x36, 0x0a, 0x31,
    0x0a, 0x32, 0x36, 0x32, 0x31, 0x35, 0x33, 0x0a, 0x37, 0x0a, 0x63, 0x6f, 0x6c, 0x75,
    0x6d, 0x6e, 0x73, 0x0a, 0x32, 0x35, 0x34, 0x0a, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x72, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x18, 0x00, 0x08, 0x00, 0x06, 0x00, 0x07, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x05, 0x14, 0x00, 0x00, 0x00, 0x48, 0x00,
    0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x73, 0x6f, 0x6d, 0x65, 0x5f, 0x63, 0x6f, 0x6c, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x07, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

alignas(8) static uint8_t kDictionaryBatch[] = {
    0xff, 0xff, 0xff, 0xff, 0xa8, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x14, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x02, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x20, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x18, 0x00,
    0x0c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x7a, 0x65, 0x72, 0x6f,
    0x6f, 0x6e, 0x65, 0x74, 0x77, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

alignas(8) static uint8_t kDictionaryRecordBatch[] = {
    0xff, 0xff, 0xff, 0xff, 0x88, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x16, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x08, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x18, 0x00, 0x0c, 0x00,
    0x04, 0x00, 0x08, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00};

static uint8_t kEndOfStream[] = {0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00};

TEST(NanoarrowIpcReader, InputStreamBuffer) {
  uint8_t input_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBuffer input;
  ArrowBufferInit(&input);
  ASSERT_EQ(ArrowBufferAppend(&input, input_data, sizeof(input_data)), NANOARROW_OK);

  struct ArrowIpcInputStream stream;
  uint8_t output_data[] = {0xff, 0xff, 0xff, 0xff, 0xff};
  int64_t size_read_bytes;

  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&stream, &input), NANOARROW_OK);
  EXPECT_EQ(input.data, nullptr);

  EXPECT_EQ(stream.read(&stream, output_data, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data1[] = {0x01, 0x02, 0xff, 0xff, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data1, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 2, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data2[] = {0x01, 0x02, 0x03, 0x04, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data2, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 4, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 1);
  uint8_t output_data3[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  EXPECT_EQ(memcmp(output_data, output_data3, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 0, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  stream.release(&stream);
}

// clang-tidy helpfully reminds us that file_ptr might not be released
// if an assertion fails
struct FileCloser {
  FileCloser(FILE* file) : file_(file) {}
  ~FileCloser() {
    if (file_) fclose(file_);
  }
  FILE* file_{};
};

TEST(NanoarrowIpcReader, InputStreamFile) {
  struct ArrowIpcInputStream stream;
  errno = EINVAL;
  ASSERT_EQ(ArrowIpcInputStreamInitFile(&stream, nullptr, 1), EINVAL);

  uint8_t input_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
#ifdef _MSC_VER
  // This macro suppresses warnings when using tmpfile and the MSVC compiler
  // warning C4996: 'tmpfile': This function or variable may be unsafe. Consider
  // using tmpfile_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS
  FILE* file_ptr;
  ASSERT_EQ(tmpfile_s(&file_ptr), 0);
#else
  FILE* file_ptr = tmpfile();
#endif
  FileCloser closer{file_ptr};
  ASSERT_NE(file_ptr, nullptr);
  ASSERT_EQ(fwrite(input_data, 1, sizeof(input_data), file_ptr), sizeof(input_data));
  fseek(file_ptr, 0, SEEK_SET);

  uint8_t output_data[] = {0xff, 0xff, 0xff, 0xff, 0xff};
  int64_t size_read_bytes;

  ASSERT_EQ(ArrowIpcInputStreamInitFile(&stream, file_ptr, /*close_on_release=*/1),
            NANOARROW_OK);
  closer.file_ = nullptr;

  EXPECT_EQ(stream.read(&stream, output_data, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data1[] = {0x01, 0x02, 0xff, 0xff, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data1, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 2, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data2[] = {0x01, 0x02, 0x03, 0x04, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data2, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 4, 2, &size_read_bytes, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 1);
  uint8_t output_data3[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  EXPECT_EQ(memcmp(output_data, output_data3, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 0, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  stream.release(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderBasic) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
      NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;

  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  ArrowArrayRelease(&array);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderBasicNoSharedBuffers) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
      NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  struct ArrowIpcArrayStreamReaderOptions options;
  options.field_index = -1;
  options.use_shared_buffers = 0;
  options.validation_level = NANOARROW_VALIDATION_LEVEL_FULL;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, &options), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  ArrowArrayRelease(&array);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderBufferInputSharedBuffers) {
  // When the input is a buffer, decoded arrays should reference it in place when
  // shared buffers are requested (and the input is aligned), and keep it alive
  // after the stream has been released.
  struct TestCase {
    int use_shared_buffers;
    bool input_is_shared;
    bool input_is_misaligned;
    bool expect_values_in_input;
  };

  alignas(8)
      uint8_t misaligned_storage[1 + sizeof(kSimpleSchema) + sizeof(kSimpleRecordBatch)];

  for (const auto& test_case :
       {TestCase{1, false, false, true}, TestCase{1, true, false, true},
        TestCase{0, false, false, false}, TestCase{0, true, false, false},
        TestCase{1, false, true, false}}) {
    SCOPED_TRACE(
        "use_shared_buffers: " + std::to_string(test_case.use_shared_buffers) +
        ", input_is_shared: " + std::to_string(test_case.input_is_shared) +
        ", input_is_misaligned: " + std::to_string(test_case.input_is_misaligned));

    struct ArrowBuffer input_buffer;
    ArrowBufferInit(&input_buffer);
    if (test_case.input_is_misaligned) {
      // Borrow storage that starts one byte past an aligned address
      uint8_t* data = misaligned_storage + 1;
      memcpy(data, kSimpleSchema, sizeof(kSimpleSchema));
      memcpy(data + sizeof(kSimpleSchema), kSimpleRecordBatch,
             sizeof(kSimpleRecordBatch));
      ASSERT_EQ(ArrowBufferSetAllocator(
                    &input_buffer,
                    ArrowBufferDeallocator(
                        [](ArrowBufferAllocator*, uint8_t*, int64_t) {}, nullptr)),
                NANOARROW_OK);
      input_buffer.data = data;
      input_buffer.size_bytes = sizeof(kSimpleSchema) + sizeof(kSimpleRecordBatch);
    } else {
      ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
                NANOARROW_OK);
      ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleRecordBatch,
                                  sizeof(kSimpleRecordBatch)),
                NANOARROW_OK);
    }

    if (test_case.input_is_shared) {
      struct ArrowBuffer shared;
      ASSERT_EQ(ArrowSharedBufferInit(&shared, &input_buffer), NANOARROW_OK);
      ArrowBufferMove(&shared, &input_buffer);
    }

    const uint8_t* input_begin = input_buffer.data;
    const uint8_t* input_end = input_buffer.data + input_buffer.size_bytes;

    struct ArrowIpcRandomAccessInput input;
    ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(&input, &input_buffer), NANOARROW_OK);

    struct ArrowArrayStream stream;
    struct ArrowIpcArrayStreamReaderOptions options;
    options.field_index = -1;
    options.use_shared_buffers = test_case.use_shared_buffers;
    options.validation_level = NANOARROW_VALIDATION_LEVEL_FULL;
    ASSERT_EQ(ArrowIpcArrayStreamReaderInitRandomAccess(&stream, &input, &options),
              NANOARROW_OK);

    struct ArrowArray array;
    ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
    ArrowArrayStreamRelease(&stream);

    ASSERT_EQ(array.n_children, 1);
    ASSERT_EQ(array.children[0]->length, 3);
    auto values = reinterpret_cast<const uint8_t*>(array.children[0]->buffers[1]);
    bool values_in_input = values >= input_begin && values < input_end;
    EXPECT_EQ(values_in_input, test_case.expect_values_in_input);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(values) % 8, 0);

    int32_t int_values[3];
    memcpy(int_values, values, sizeof(int_values));
    EXPECT_EQ(int_values[0], 1);
    EXPECT_EQ(int_values[1], 2);
    EXPECT_EQ(int_values[2], 3);

    ArrowArrayRelease(&array);
  }
}

TEST(NanoarrowIpcReader, StreamReaderBasicWithEndOfStream) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
      NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kEndOfStream, sizeof(kEndOfStream)),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  ArrowArrayRelease(&array);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderIncompleteMessageHeader) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema) - 1),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected >= 280 bytes of remaining data but found 279 bytes in buffer");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderIncompleteMessageBody) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  // Truncate the record batch at the very end of the body
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleRecordBatch,
                              sizeof(kSimpleRecordBatch) - 1),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), ESPIPE);
  EXPECT_STREQ(stream.get_last_error(&stream),
               "Expected to be able to read 16 bytes for message body but got 15");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderExpectedRecordBatch) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Unexpected message type (expected RecordBatch or DictionaryBatch)");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderExpectedSchema) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
      NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Unexpected message type at start of input (expected Schema)");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcTest, StreamReaderInvalidBuffer) {
  struct ArrowBuffer input_buffer;
  struct ArrowIpcInputStream input;
  struct ArrowArrayStream stream;
  struct ArrowSchema schema;
  struct ArrowArray array;

  uint8_t simple_stream_invalid[sizeof(kSimpleSchema) + sizeof(kSimpleRecordBatch)];
  struct ArrowBufferView data;
  data.data.as_uint8 = simple_stream_invalid;

  // Create invalid data by removing bytes one at a time and ensuring an error code and
  // a null-terminated error. After byte 273/280 this passes because the bytes are just
  // padding.
  data.size_bytes = sizeof(kSimpleSchema);
  for (int64_t i = 1; i < 273; i++) {
    SCOPED_TRACE(i);

    memcpy(simple_stream_invalid, kSimpleSchema, i);
    memcpy(simple_stream_invalid + i, kSimpleSchema + (i + 1),
           (sizeof(kSimpleSchema) - i - 1));

    ArrowBufferInit(&input_buffer);
    ASSERT_EQ(ArrowBufferAppendBufferView(&input_buffer, data), NANOARROW_OK);
    ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);
    ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

    ASSERT_NE(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
    ASSERT_GT(strlen(ArrowArrayStreamGetLastError(&stream)), 0);

    ArrowArrayStreamRelease(&stream);
  }

  // Do the same exercise removing bytes of the record batch message.
  // Similarly, this succeeds if the byte removed is part of the padding at the end.
  memcpy(simple_stream_invalid, kSimpleSchema, sizeof(kSimpleSchema));
  data.size_bytes = sizeof(simple_stream_invalid);
  for (int64_t i = 1; i < 144; i++) {
    SCOPED_TRACE(i);

    memcpy(simple_stream_invalid + sizeof(kSimpleSchema), kSimpleRecordBatch, i);
    memcpy(simple_stream_invalid + sizeof(kSimpleSchema) + i,
           kSimpleRecordBatch + (i + 1), (sizeof(kSimpleRecordBatch) - i - 1));

    ArrowBufferInit(&input_buffer);
    ASSERT_EQ(ArrowBufferAppendBufferView(&input_buffer, data), NANOARROW_OK);
    ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);
    ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
    ArrowSchemaRelease(&schema);
    ASSERT_NE(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
    ASSERT_GT(strlen(ArrowArrayStreamGetLastError(&stream)), 0);

    ArrowArrayStreamRelease(&stream);
  }
}

TEST(NanoarrowIpcReader, StreamReaderUnsupportedFieldIndex) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
      NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  struct ArrowIpcArrayStreamReaderOptions options;
  options.field_index = 0;
  options.use_shared_buffers = 0;
  options.validation_level = NANOARROW_VALIDATION_LEVEL_FULL;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, &options), NANOARROW_OK);

  struct ArrowSchema schema;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, &error), ENOTSUP);
  EXPECT_STREQ(error.message, "Field index != -1 is not yet supported");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderEmptyInput) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, &error), ENODATA);
  EXPECT_STREQ(error.message, "No data available on stream");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderIncompletePrefix) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppendUInt8(&input_buffer, 0x00), NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, &error), EINVAL);
  EXPECT_STREQ(error.message, "Expected at least 8 bytes in remainder of stream");

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderDictionary) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kDictionarySchema, sizeof(kDictionarySchema)),
      NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kDictionaryBatch, sizeof(kDictionaryBatch)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kDictionaryRecordBatch,
                              sizeof(kDictionaryRecordBatch)),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ASSERT_EQ(schema.n_children, 1);
  // Dictionary-encoded field with int8 indices
  EXPECT_STREQ(schema.children[0]->format, "c");
  ASSERT_NE(schema.children[0]->dictionary, nullptr);
  // Dictionary values are utf8 strings
  EXPECT_STREQ(schema.children[0]->dictionary->format, "u");
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  ASSERT_EQ(array.n_children, 1);
  // The child should have a dictionary
  ASSERT_NE(array.children[0]->dictionary, nullptr);
  EXPECT_EQ(array.children[0]->dictionary->length, 3);
  ArrowArrayRelease(&array);

  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderValidationLevel) {
  // Point the last dictionary index past the end of the three-value dictionary.
  // Only full validation inspects index values, so a lower level accepts the batch.
  uint8_t record_batch[sizeof(kDictionaryRecordBatch)];
  memcpy(record_batch, kDictionaryRecordBatch, sizeof(kDictionaryRecordBatch));
  ASSERT_EQ(record_batch[sizeof(record_batch) - 6], 0x02);
  record_batch[sizeof(record_batch) - 6] = 0x07;

  for (auto validation_level :
       {NANOARROW_VALIDATION_LEVEL_FULL, NANOARROW_VALIDATION_LEVEL_DEFAULT}) {
    SCOPED_TRACE(validation_level);
    struct ArrowBuffer input_buffer;
    ArrowBufferInit(&input_buffer);
    ASSERT_EQ(
        ArrowBufferAppend(&input_buffer, kDictionarySchema, sizeof(kDictionarySchema)),
        NANOARROW_OK);
    ASSERT_EQ(
        ArrowBufferAppend(&input_buffer, kDictionaryBatch, sizeof(kDictionaryBatch)),
        NANOARROW_OK);
    ASSERT_EQ(ArrowBufferAppend(&input_buffer, record_batch, sizeof(record_batch)),
              NANOARROW_OK);

    struct ArrowIpcInputStream input;
    ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

    struct ArrowArrayStream stream;
    struct ArrowIpcArrayStreamReaderOptions options;
    options.field_index = -1;
    options.use_shared_buffers = ArrowSharedBufferIsThreadSafe();
    options.validation_level = validation_level;
    ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, &options), NANOARROW_OK);

    struct ArrowArray array;
    if (validation_level == NANOARROW_VALIDATION_LEVEL_FULL) {
      ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), EINVAL);
      EXPECT_NE(std::string(ArrowArrayStreamGetLastError(&stream))
                    .find("Expected dictionary index >= 0 and < 3 but found value 7"),
                std::string::npos);
    } else {
      ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, nullptr), NANOARROW_OK);
      EXPECT_EQ(array.length, 3);
      ArrowArrayRelease(&array);
    }

    ArrowArrayStreamRelease(&stream);
  }
}

TEST(NanoarrowIpcReader, StreamReaderInvalidValidationLevel) {
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  struct ArrowIpcArrayStreamReaderOptions options;
  options.field_index = -1;
  options.use_shared_buffers = 0;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange): deliberately invalid
  options.validation_level = static_cast<ArrowValidationLevel>(4);
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, &options), EINVAL);

  input.release(&input);
}

TEST(NanoarrowIpcReader, StreamReaderDictionaryBatchWithoutDictionarySchema) {
  // Send a dictionary batch when the schema has no dictionaries
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kDictionaryBatch, sizeof(kDictionaryBatch)),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  struct ArrowError error;
  ASSERT_NE(ArrowArrayStreamGetNext(&stream, &array, &error), NANOARROW_OK);
  ASSERT_GT(strlen(ArrowArrayStreamGetLastError(&stream)), 0);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, StreamReaderRecordBatchWithoutDictionaryBatch) {
  // Send a record batch referencing a dictionary before the dictionary values arrive
  struct ArrowBuffer input_buffer;
  ArrowBufferInit(&input_buffer);
  ASSERT_EQ(
      ArrowBufferAppend(&input_buffer, kDictionarySchema, sizeof(kDictionarySchema)),
      NANOARROW_OK);
  // Skip the dictionary batch and go straight to the record batch
  ASSERT_EQ(ArrowBufferAppend(&input_buffer, kDictionaryRecordBatch,
                              sizeof(kDictionaryRecordBatch)),
            NANOARROW_OK);

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, &input_buffer), NANOARROW_OK);

  struct ArrowArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(&stream, &input, nullptr), NANOARROW_OK);

  struct ArrowSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(&stream, &schema, nullptr), NANOARROW_OK);
  ArrowSchemaRelease(&schema);

  struct ArrowArray array;
  struct ArrowError error;
  // Should error because dictionary values were never provided
  ASSERT_EQ(ArrowArrayStreamGetNext(&stream, &array, &error), EINVAL);
  ASSERT_GT(strlen(ArrowArrayStreamGetLastError(&stream)), 0);

  ArrowArrayStreamRelease(&stream);
}

TEST(NanoarrowIpcReader, RandomAccessBufferBoundsAndLifetime) {
  nanoarrow::UniqueBuffer buffer;
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), "abcde", 5), NANOARROW_OK);
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(input.get(), buffer.get()), NANOARROW_OK);
  EXPECT_EQ(buffer->data, nullptr);

  ArrowBufferView peek;
  ASSERT_EQ(input->peek_at(input.get(), 2, 100, &peek, nullptr), NANOARROW_OK);
  ASSERT_EQ(peek.size_bytes, 3);
  EXPECT_EQ(memcmp(peek.data.data, "cde", 3), 0);
  EXPECT_EQ(input->peek_at(input.get(), -1, 1, &peek, nullptr), EINVAL);
  EXPECT_EQ(input->peek_at(input.get(), 0, -1, &peek, nullptr), EINVAL);
  ASSERT_EQ(input->peek_at(input.get(), INT64_MAX, INT64_MAX, &peek, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(peek.size_bytes, 0);

  ASSERT_EQ(input->read_at(input.get(), 3, 100, buffer.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(buffer->size_bytes, 2);
  EXPECT_EQ(memcmp(buffer->data, "de", 2), 0);
  // Absolute offsets allow going backwards, independent of previous reads.
  ASSERT_EQ(input->read_at(input.get(), 0, 2, buffer.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(buffer->size_bytes, 2);
  EXPECT_EQ(memcmp(buffer->data, "ab", 2), 0);
  ASSERT_EQ(input->read_at(input.get(), 0, 0, buffer.get(), nullptr), NANOARROW_OK);
  EXPECT_EQ(buffer->size_bytes, 0);
  ASSERT_EQ(input->read_at(input.get(), 1, 3, buffer.get(), nullptr), NANOARROW_OK);
  nanoarrow::ipc::UniqueRandomAccessInput moved(std::move(input));
  EXPECT_EQ(input->release, nullptr);
  moved.reset();
  EXPECT_EQ(memcmp(buffer->data, "bcd", 3), 0);
}

TEST(NanoarrowIpcReader, RandomAccessFileBoundsAndSeeking) {
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);
  ASSERT_EQ(fwrite("abcde", 1, 5, file), 5);
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitFile(input.get(), nullptr, 0), EINVAL);
  ASSERT_EQ(ArrowIpcRandomAccessInputInitFile(input.get(), file, 0), NANOARROW_OK);
  ArrowBufferView peek;
  EXPECT_EQ(input->peek_at(input.get(), 0, 1, &peek, nullptr), ENOTSUP);
  nanoarrow::UniqueBuffer out;
  EXPECT_EQ(input->read_at(input.get(), -1, 1, out.get(), nullptr), EINVAL);
  EXPECT_EQ(input->read_at(input.get(), 0, -1, out.get(), nullptr), EINVAL);
  ASSERT_EQ(input->read_at(input.get(), 3, 10, out.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(out->size_bytes, 2);
  EXPECT_EQ(memcmp(out->data, "de", 2), 0);
  ASSERT_EQ(input->read_at(input.get(), 5, 10, out.get(), nullptr), NANOARROW_OK);
  EXPECT_EQ(out->size_bytes, 0);
  ASSERT_EQ(input->read_at(input.get(), 1, 3, out.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(out->size_bytes, 3);
  EXPECT_EQ(memcmp(out->data, "bcd", 3), 0);
  input.reset();
  EXPECT_EQ(fseek(file, 0, SEEK_SET), 0);
  EXPECT_EQ(fclose(file), 0);
  EXPECT_EQ(memcmp(out->data, "bcd", 3), 0);
}

// A custom input whose peek result is invalidated by its next read. This also
// exercises the reader without relying on the built-in callback addresses.
struct RandomAccessReaderTestInput {
  nanoarrow::ipc::UniqueRandomAccessInput underlying;
  nanoarrow::UniqueBuffer scratch;
  int read_count = 0;
  int peek_count = 0;
  bool peek_unsupported = false;
};

static void InitRandomAccessReaderTestInput(RandomAccessReaderTestInput* state,
                                            ArrowIpcRandomAccessInput* input) {
  input->private_data = state;
  input->peek_at = [](ArrowIpcRandomAccessInput* self, int64_t offset, int64_t size,
                      ArrowBufferView* out, ArrowError* error) -> ArrowErrorCode {
    auto* state = static_cast<RandomAccessReaderTestInput*>(self->private_data);
    state->peek_count++;
    if (state->peek_unsupported) {
      return ENOTSUP;
    }
    NANOARROW_RETURN_NOT_OK(state->underlying->read_at(
        state->underlying.get(), offset, size, state->scratch.get(), error));
    // Make a private copy that the next callback can overwrite.
    nanoarrow::UniqueBuffer copy;
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppend(copy.get(), state->scratch->data, state->scratch->size_bytes));
    state->scratch = std::move(copy);
    out->data.data = state->scratch->data;
    out->size_bytes = state->scratch->size_bytes;
    return NANOARROW_OK;
  };
  input->read_at = [](ArrowIpcRandomAccessInput* self, int64_t offset, int64_t size,
                      ArrowBuffer* out, ArrowError* error) -> ArrowErrorCode {
    auto* state = static_cast<RandomAccessReaderTestInput*>(self->private_data);
    state->read_count++;
    if (state->scratch->size_bytes > 0) {
      memset(state->scratch->data, 0xff, state->scratch->size_bytes);
    }
    return state->underlying->read_at(state->underlying.get(), offset, size, out, error);
  };
  input->release = [](ArrowIpcRandomAccessInput* self) {
    auto* state = static_cast<RandomAccessReaderTestInput*>(self->private_data);
    state->underlying.reset();
    state->scratch.reset();
    self->release = nullptr;
  };
}

TEST(NanoarrowIpcReader, RandomAccessReaderCustomAndFileInputs) {
  for (int backend : {0, 1, 2, 3}) {
    for (int sharing : {0, 1}) {
      SCOPED_TRACE("backend=" + std::to_string(backend) +
                   " sharing=" + std::to_string(sharing));
      nanoarrow::UniqueBuffer buffer;
      ASSERT_EQ(ArrowBufferAppend(buffer.get(), kSimpleSchema, sizeof(kSimpleSchema)),
                NANOARROW_OK);
      ASSERT_EQ(
          ArrowBufferAppend(buffer.get(), kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
          NANOARROW_OK);
      ASSERT_EQ(
          ArrowBufferAppend(buffer.get(), kSimpleRecordBatch, sizeof(kSimpleRecordBatch)),
          NANOARROW_OK);
      ASSERT_EQ(ArrowBufferAppend(buffer.get(), kEndOfStream, sizeof(kEndOfStream)),
                NANOARROW_OK);
      RandomAccessReaderTestInput state;
      if (backend == 3) {
        FILE* file = tmpfile();
        ASSERT_NE(file, nullptr);
        ASSERT_EQ(fwrite(buffer->data, 1, buffer->size_bytes, file), buffer->size_bytes);
        ASSERT_EQ(ArrowIpcRandomAccessInputInitFile(state.underlying.get(), file, 1),
                  NANOARROW_OK);
      } else {
        ASSERT_EQ(
            ArrowIpcRandomAccessInputInitBuffer(state.underlying.get(), buffer.get()),
            NANOARROW_OK);
      }
      nanoarrow::ipc::UniqueRandomAccessInput input;
      InitRandomAccessReaderTestInput(&state, input.get());
      if (backend == 1) input->peek_at = nullptr;
      if (backend == 2) state.peek_unsupported = true;
      ArrowIpcArrayStreamReaderOptions options = {-1, sharing,
                                                  NANOARROW_VALIDATION_LEVEL_FULL};
      nanoarrow::UniqueArrayStream reader;
      ASSERT_EQ(
          ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
          NANOARROW_OK);
      EXPECT_EQ(input->release, nullptr);
      nanoarrow::UniqueSchema schema;
      ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), schema.get(), nullptr),
                NANOARROW_OK);
      nanoarrow::UniqueArray first, second, eos;
      ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), first.get(), nullptr),
                NANOARROW_OK);
      ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), second.get(), nullptr),
                NANOARROW_OK);
      ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), eos.get(), nullptr), NANOARROW_OK);
      EXPECT_EQ(eos->release, nullptr);
      reader.reset();
      EXPECT_GT(state.read_count, 0);
      if (backend != 1) EXPECT_GT(state.peek_count, 0);
      for (auto* array : {first.get(), second.get()}) {
        ASSERT_EQ(array->length, 3);
        const auto* values = static_cast<const int32_t*>(array->children[0]->buffers[1]);
        EXPECT_EQ(values[0], 1);
        EXPECT_EQ(values[1], 2);
        EXPECT_EQ(values[2], 3);
      }
    }
  }
}

TEST(NanoarrowIpcReader, RandomAccessReaderUnalignedBufferOffset) {
  // The body is aligned, but its int32 buffer starts at offset one.
  uint8_t batch[sizeof(kSimpleRecordBatch)];
  memcpy(batch, kSimpleRecordBatch, sizeof(batch));
  ASSERT_EQ(batch[104], 0);
  batch[104] = 1;  // Second Buffer.offset in the flatbuffer header.
  memmove(batch + 145, batch + 144, 12);
  nanoarrow::UniqueBuffer buffer;
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), batch, sizeof(batch)), NANOARROW_OK);
  const uint8_t* original_values = buffer->data + sizeof(kSimpleSchema) + 145;
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(input.get(), buffer.get()), NANOARROW_OK);
  ArrowIpcArrayStreamReaderOptions options = {-1, 1, NANOARROW_VALIDATION_LEVEL_FULL};
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(
      ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
      NANOARROW_OK);
  nanoarrow::UniqueArray array;
  ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  const auto* values = static_cast<const int32_t*>(array->children[0]->buffers[1]);
  EXPECT_NE(static_cast<const void*>(values), original_values);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(values) % alignof(int32_t), 0);
  reader.reset();
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 3);
}

TEST(NanoarrowIpcReader, RandomAccessReaderDictionaryLifetime) {
  nanoarrow::UniqueBuffer buffer;
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kDictionarySchema, sizeof(kDictionarySchema)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kDictionaryBatch, sizeof(kDictionaryBatch)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kDictionaryRecordBatch,
                              sizeof(kDictionaryRecordBatch)),
            NANOARROW_OK);
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(input.get(), buffer.get()), NANOARROW_OK);
  ArrowIpcArrayStreamReaderOptions options = {-1, 1, NANOARROW_VALIDATION_LEVEL_FULL};
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(
      ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
      NANOARROW_OK);
  nanoarrow::UniqueArray array;
  ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  reader.reset();
  ASSERT_NE(array->children[0]->dictionary, nullptr);
  const auto* dictionary = array->children[0]->dictionary;
  ASSERT_EQ(dictionary->length, 3);
  EXPECT_EQ(memcmp(dictionary->buffers[2], "zeroonetwo", 10), 0);
}

TEST(NanoarrowIpcReader, RandomAccessReaderTruncatedBodyAndInvalidOptions) {
  nanoarrow::UniqueBuffer buffer;
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kSimpleSchema, sizeof(kSimpleSchema)),
            NANOARROW_OK);
  ASSERT_EQ(
      ArrowBufferAppend(buffer.get(), kSimpleRecordBatch, sizeof(kSimpleRecordBatch) - 1),
      NANOARROW_OK);
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(input.get(), buffer.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ArrowIpcArrayStreamReaderOptions options = {-1, 1,
                                              static_cast<ArrowValidationLevel>(999)};
  EXPECT_EQ(
      ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
      EINVAL);
  EXPECT_NE(input->release, nullptr);
  options.validation_level = NANOARROW_VALIDATION_LEVEL_FULL;
  ASSERT_EQ(
      ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
      NANOARROW_OK);
  nanoarrow::UniqueArray array;
  ArrowError error;
  EXPECT_EQ(ArrowArrayStreamGetNext(reader.get(), array.get(), &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected to be able to read 16 bytes for message body but got 15");
}

TEST(NanoarrowIpcReader, RandomAccessReaderCallbackErrors) {
  for (int mode : {0, 1, 2, 3}) {
    SCOPED_TRACE(mode);
    ArrowIpcRandomAccessInput input = {};
    input.private_data = &mode;
    input.release = [](ArrowIpcRandomAccessInput* self) { self->release = nullptr; };
    input.read_at = [](ArrowIpcRandomAccessInput* self, int64_t, int64_t size,
                       ArrowBuffer* out, ArrowError* error) -> ArrowErrorCode {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(out, "x", 1));
      if (*static_cast<int*>(self->private_data) == 2) {
        ArrowErrorSet(error, "read failure");
        return EIO;
      }
      out->size_bytes = size + 1;
      return NANOARROW_OK;
    };
    if (mode < 2) {
      input.peek_at = [](ArrowIpcRandomAccessInput* self, int64_t, int64_t size,
                         ArrowBufferView* out, ArrowError* error) -> ArrowErrorCode {
        if (*static_cast<int*>(self->private_data) == 0) {
          ArrowErrorSet(error, "peek failure");
          return EIO;
        }
        out->data.data = nullptr;
        out->size_bytes = size + 1;
        return NANOARROW_OK;
      };
    }
    nanoarrow::UniqueArrayStream reader;
    ASSERT_EQ(ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), &input, nullptr),
              NANOARROW_OK);
    nanoarrow::UniqueSchema schema;
    ArrowError error;
    EXPECT_EQ(ArrowArrayStreamGetSchema(reader.get(), schema.get(), &error),
              mode % 2 == 0 ? EIO : EINVAL);
    EXPECT_GT(strlen(error.message), 0);
  }
  ArrowIpcRandomAccessInput invalid = {};
  nanoarrow::UniqueArrayStream reader;
  EXPECT_EQ(ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), &invalid, nullptr),
            EINVAL);
}

TEST(NanoarrowIpcReader, RandomAccessReaderLegacyHeaders) {
  nanoarrow::UniqueBuffer buffer;
  for (ArrowBufferView message :
       {ArrowBufferView{{kSimpleSchema}, sizeof(kSimpleSchema)},
        ArrowBufferView{{kSimpleRecordBatch}, sizeof(kSimpleRecordBatch)}}) {
    nanoarrow::UniqueBuffer legacy;
    ASSERT_EQ(ArrowBufferAppendBufferView(legacy.get(), message), NANOARROW_OK);
    // Locate Message.version through the flatbuffer's root and vtable, then set V4.
    auto read16 = [](const uint8_t* p) { return p[0] | (p[1] << 8); };
    auto read32 = [](const uint8_t* p) {
      return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    };
    int table = 8 + read32(legacy->data + 8);
    int vtable = table - read32(legacy->data + table);
    int version = table + read16(legacy->data + vtable + 4);
    legacy->data[version] = 3;  // MetadataVersion.V4
    legacy->data[version + 1] = 0;
    // Legacy messages omit the continuation token.
    ASSERT_EQ(ArrowBufferAppend(buffer.get(), legacy->data + 4, legacy->size_bytes - 4),
              NANOARROW_OK);
  }
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), kEndOfStream + 4, 4), NANOARROW_OK);
  nanoarrow::ipc::UniqueRandomAccessInput input;
  ASSERT_EQ(ArrowIpcRandomAccessInputInitBuffer(input.get(), buffer.get()), NANOARROW_OK);
  ArrowIpcArrayStreamReaderOptions options = {-1, 1, NANOARROW_VALIDATION_LEVEL_FULL};
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(
      ArrowIpcArrayStreamReaderInitRandomAccess(reader.get(), input.get(), &options),
      NANOARROW_OK);
  nanoarrow::UniqueArray array, eos;
  ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(array->length, 3);
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), eos.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(eos->release, nullptr);
  reader.reset();
  EXPECT_EQ(static_cast<const int32_t*>(array->children[0]->buffers[1])[2], 3);
}
