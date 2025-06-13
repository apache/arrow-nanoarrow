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

#include "nanoarrow/nanoarrow_ipc.h"

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
  EXPECT_STREQ(error.message, "Unexpected message type (expected RecordBatch)");

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
