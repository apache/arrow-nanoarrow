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

#include <stdexcept>

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/ipc/api.h>
#include <arrow/util/key_value_metadata.h>
#include <gtest/gtest.h>

// For bswap32()
#include "flatcc/portable/pendian.h"

#include "nanoarrow_ipc.h"

using namespace arrow;

// Copied from nanoarrow_ipc.c so we can test the internal state
// of the decoder
extern "C" {
struct ArrowIpcField {
  struct ArrowArrayView* array_view;
  int64_t buffer_offset;
};

struct ArrowIpcDecoderPrivate {
  enum ArrowIpcEndianness endianness;
  enum ArrowIpcEndianness system_endianness;
  struct ArrowArrayView array_view;
  int64_t n_fields;
  struct ArrowIpcField* fields;
  int64_t n_buffers;
  const void* last_message;
};
}

static enum ArrowIpcEndianness ArrowIpcSystemEndianness(void) {
  uint32_t check = 1;
  char first_byte;
  enum ArrowIpcEndianness system_endianness;
  memcpy(&first_byte, &check, sizeof(char));
  if (first_byte) {
    return NANOARROW_IPC_ENDIANNESS_LITTLE;
  } else {
    return NANOARROW_IPC_ENDIANNESS_BIG;
  }
}

TEST(NanoarrowIpcCheckRuntime, CheckRuntime) {
  EXPECT_EQ(ArrowIpcCheckRuntime(nullptr), NANOARROW_OK);
}

// library(arrow, warn.conflicts = FALSE)

// # R package doesn't do field metadata yet, so this hack is needed
// field <- narrow::narrow_schema("i", "some_col", metadata = list("some_key_field" =
// "some_value_field")) field <- Field$import_from_c(field)

// schema <- arrow::schema(field)
// schema$metadata <- list("some_key" = "some_value")
// schema$serialize()
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

TEST(NanoarrowIpcTest, NanoarrowIpcCheckHeader) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  uint32_t negative_one_le = static_cast<uint32_t>(-1);
  uint32_t one_le = 1;
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    negative_one_le = bswap32(negative_one_le);
    one_le = bswap32(one_le);
  }

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = 1;

  ArrowIpcDecoderInit(&decoder);

  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected data of at least 8 bytes but only 1 bytes remain");

  uint32_t eight_bad_bytes[] = {0, 0};
  data.data.as_uint8 = reinterpret_cast<uint8_t*>(eight_bad_bytes);
  data.size_bytes = 8;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0xFFFFFFFF at start of message but found 0x00000000");

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = negative_one_le;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected message body size > 0 but found message body size of -1 bytes");

  eight_bad_bytes[1] = one_le;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);

  EXPECT_STREQ(error.message,
               "Expected 0 <= message body size <= 0 bytes but found message body size "
               "of 1 bytes");

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = 0;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ENODATA);
  EXPECT_STREQ(error.message, "End of Arrow stream");

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcPeekSimpleSchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcDecoderInit(&decoder);
  EXPECT_EQ(ArrowIpcDecoderPeekHeader(&decoder, data, &error), NANOARROW_OK);
  EXPECT_EQ(decoder.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(decoder.body_size_bytes, 0);

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifySimpleSchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcDecoderInit(&decoder);
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), NANOARROW_OK);
  EXPECT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);
  EXPECT_EQ(decoder.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(decoder.body_size_bytes, 0);

  uint8_t simple_schema_invalid[280];
  memcpy(simple_schema_invalid, kSimpleSchema, sizeof(simple_schema_invalid));
  memset(simple_schema_invalid + 8, 0xFF, sizeof(simple_schema_invalid) - 8);

  data.data.as_uint8 = simple_schema_invalid;
  data.size_bytes = sizeof(kSimpleSchema);
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Message flatbuffer verification failed");

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifySimpleRecordBatch) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleRecordBatch;
  data.size_bytes = sizeof(kSimpleRecordBatch);

  ArrowIpcDecoderInit(&decoder);
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), NANOARROW_OK);
  EXPECT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);
  EXPECT_EQ(decoder.header_size_bytes,
            sizeof(kSimpleRecordBatch) - decoder.body_size_bytes);
  EXPECT_EQ(decoder.body_size_bytes, 16);

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleSchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcDecoderInit(&decoder);

  EXPECT_EQ(ArrowIpcDecoderDecodeSchema(&decoder, &schema, &error), EINVAL);
  EXPECT_STREQ(error.message, "decoder did not just decode a Schema message");

  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), NANOARROW_OK);
  EXPECT_EQ(decoder.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(decoder.body_size_bytes, 0);

  EXPECT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);
  EXPECT_EQ(decoder.endianness, NANOARROW_IPC_ENDIANNESS_LITTLE);
  EXPECT_EQ(decoder.feature_flags, 0);

  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(&decoder, &schema, &error), NANOARROW_OK);
  ASSERT_EQ(schema.n_children, 1);
  EXPECT_STREQ(schema.children[0]->name, "some_col");
  EXPECT_EQ(schema.children[0]->flags, ARROW_FLAG_NULLABLE);
  EXPECT_STREQ(schema.children[0]->format, "i");

  schema.release(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleRecordBatch) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;
  struct ArrowArray array;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeStruct(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleRecordBatch;
  data.size_bytes = sizeof(kSimpleRecordBatch);

  ArrowIpcDecoderInit(&decoder);
  auto decoder_private =
      reinterpret_cast<struct ArrowIpcDecoderPrivate*>(decoder.private_data);

  // Attempt to get array should fail nicely here
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, data, 0, nullptr, &error), EINVAL);
  EXPECT_STREQ(error.message, "decoder did not just decode a RecordBatch message");

  ASSERT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), NANOARROW_OK);
  EXPECT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);
  EXPECT_EQ(decoder.header_size_bytes,
            sizeof(kSimpleRecordBatch) - decoder.body_size_bytes);
  EXPECT_EQ(decoder.body_size_bytes, 16);

  EXPECT_EQ(decoder.codec, NANOARROW_IPC_COMPRESSION_TYPE_NONE);

  struct ArrowBufferView body;
  body.data.as_uint8 = kSimpleRecordBatch + decoder.header_size_bytes;
  body.size_bytes = decoder.body_size_bytes;

  // Check full struct extract
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, -1, &array, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  ASSERT_EQ(array.n_children, 1);
  ASSERT_EQ(array.children[0]->n_buffers, 2);
  ASSERT_EQ(array.children[0]->length, 3);
  EXPECT_EQ(array.children[0]->null_count, 0);
  const int32_t* out = reinterpret_cast<const int32_t*>(array.children[0]->buffers[1]);

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_LITTLE) {
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 3);
  } else {
    EXPECT_EQ(out[0], bswap32(1));
    EXPECT_EQ(out[1], bswap32(2));
    EXPECT_EQ(out[2], bswap32(3));
  }

  array.release(&array);

  // Check field extract
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array, nullptr), NANOARROW_OK);
  ASSERT_EQ(array.n_buffers, 2);
  ASSERT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  out = reinterpret_cast<const int32_t*>(array.buffers[1]);

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_LITTLE) {
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(out[1], 2);
    EXPECT_EQ(out[2], 3);
  } else {
    EXPECT_EQ(out[0], bswap32(1));
    EXPECT_EQ(out[1], bswap32(2));
    EXPECT_EQ(out[2], bswap32(3));
  }

  array.release(&array);

  // Field extract should fail if compression was set
  decoder.codec = NANOARROW_IPC_COMPRESSION_TYPE_ZSTD;
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array, &error), ENOTSUP);
  EXPECT_STREQ(error.message, "The nanoarrow_ipc extension does not support compression");
  decoder.codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;

  // Field extract should fail on non-system endian
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_LITTLE) {
    ArrowIpcDecoderSetEndianness(&decoder, NANOARROW_IPC_ENDIANNESS_BIG);
  } else {
    ArrowIpcDecoderSetEndianness(&decoder, NANOARROW_IPC_ENDIANNESS_LITTLE);
  }
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array, &error), ENOTSUP);
  EXPECT_STREQ(error.message,
               "The nanoarrow_ipc extension does not support non-system endianness");
  ArrowIpcDecoderSetEndianness(&decoder, NANOARROW_IPC_ENDIANNESS_UNINITIALIZED);

  // Field extract should fail if body is too small
  body.size_bytes = 0;
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Buffer 1 requires body offsets [0..12) but body has size 0");

  // Should error if the number of buffers or field nodes doesn't match
  // (different numbers because we count the root struct and the message does not)
  decoder_private->n_buffers = 1;
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Expected 0 buffers in message but found 2");

  decoder_private->n_fields = 1;
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Expected 0 field nodes in message but found 1");

  schema.release(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcSetSchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowSchema schema;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeStruct(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "col1"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  ArrowIpcDecoderInit(&decoder);
  auto decoder_private =
      reinterpret_cast<struct ArrowIpcDecoderPrivate*>(decoder.private_data);

  EXPECT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(decoder_private->n_fields, 2);
  EXPECT_EQ(decoder_private->n_buffers, 3);

  EXPECT_EQ(decoder_private->fields[0].array_view->storage_type, NANOARROW_TYPE_STRUCT);
  EXPECT_EQ(decoder_private->fields[0].buffer_offset, 0);

  EXPECT_EQ(decoder_private->fields[1].array_view->storage_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(decoder_private->fields[1].buffer_offset, 1);

  schema.release(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcSetSchemaErrors) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;

  ArrowIpcDecoderInit(&decoder);
  ArrowSchemaInit(&schema);

  EXPECT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, &error), EINVAL);
  EXPECT_STREQ(
      error.message,
      "Error parsing schema->format: Expected a null-terminated string but found NULL");

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, &error), EINVAL);
  EXPECT_STREQ(error.message, "schema must be a struct type");

  schema.release(&schema);
  ArrowIpcDecoderReset(&decoder);
}

class ArrowTypeParameterizedTestFixture
    : public ::testing::TestWithParam<std::shared_ptr<arrow::DataType>> {
 protected:
  std::shared_ptr<arrow::DataType> data_type;
};

TEST_P(ArrowTypeParameterizedTestFixture, NanoarrowIpcArrowTypeRoundtrip) {
  const std::shared_ptr<arrow::DataType>& data_type = GetParam();
  std::shared_ptr<arrow::Schema> dummy_schema =
      arrow::schema({arrow::field("dummy_name", data_type)});
  auto maybe_serialized = arrow::ipc::SerializeSchema(*dummy_schema);
  ASSERT_TRUE(maybe_serialized.ok());

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = maybe_serialized.ValueUnsafe()->data();
  buffer_view.size_bytes = maybe_serialized.ValueOrDie()->size();

  struct ArrowIpcDecoder decoder;
  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  EXPECT_EQ(decoder.header_size_bytes, buffer_view.size_bytes);
  EXPECT_EQ(decoder.body_size_bytes, 0);

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  struct ArrowSchema schema;
  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  auto maybe_schema = arrow::ImportSchema(&schema);
  ASSERT_TRUE(maybe_schema.ok());

  // Better failure message if we first check for string equality
  EXPECT_EQ(maybe_schema.ValueUnsafe()->ToString(), dummy_schema->ToString());
  EXPECT_TRUE(maybe_schema.ValueUnsafe()->Equals(dummy_schema, true));

  ArrowIpcDecoderReset(&decoder);
}

TEST_P(ArrowTypeParameterizedTestFixture, NanoarrowIpcArrowArrayRoundtrip) {
  const std::shared_ptr<arrow::DataType>& data_type = GetParam();
  std::shared_ptr<arrow::Schema> dummy_schema =
      arrow::schema({arrow::field("dummy_name", data_type)});

  auto maybe_empty = arrow::RecordBatch::MakeEmpty(dummy_schema);
  ASSERT_TRUE(maybe_empty.ok());
  auto empty = maybe_empty.ValueUnsafe();

  auto maybe_nulls_array = arrow::MakeArrayOfNull(data_type, 3);
  ASSERT_TRUE(maybe_nulls_array.ok());
  auto nulls =
      arrow::RecordBatch::Make(dummy_schema, 3, {maybe_nulls_array.ValueUnsafe()});

  auto options = arrow::ipc::IpcWriteOptions::Defaults();

  struct ArrowSchema schema;
  struct ArrowIpcDecoder decoder;
  struct ArrowBufferView buffer_view;
  struct ArrowArray array;

  // Initialize the decoder
  ASSERT_TRUE(arrow::ExportSchema(*dummy_schema, &schema).ok());
  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);

  // Check the empty array
  auto maybe_serialized = arrow::ipc::SerializeRecordBatch(*empty, options);
  ASSERT_TRUE(maybe_serialized.ok());
  buffer_view.data.data = maybe_serialized.ValueUnsafe()->data();
  buffer_view.size_bytes = maybe_serialized.ValueOrDie()->size();

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  buffer_view.data.as_uint8 += decoder.header_size_bytes;
  buffer_view.size_bytes -= decoder.header_size_bytes;
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(&decoder, buffer_view, -1, &array, nullptr),
            NANOARROW_OK);

  auto maybe_batch = arrow::ImportRecordBatch(&array, dummy_schema);
  ASSERT_TRUE(maybe_batch.ok());
  EXPECT_EQ(maybe_batch.ValueUnsafe()->ToString(), empty->ToString());
  EXPECT_TRUE(maybe_batch.ValueUnsafe()->Equals(*empty));

  // Check the array with 3 null values
  maybe_serialized = arrow::ipc::SerializeRecordBatch(*nulls, options);
  ASSERT_TRUE(maybe_serialized.ok());
  buffer_view.data.data = maybe_serialized.ValueUnsafe()->data();
  buffer_view.size_bytes = maybe_serialized.ValueOrDie()->size();

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  buffer_view.data.as_uint8 += decoder.header_size_bytes;
  buffer_view.size_bytes -= decoder.header_size_bytes;
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(&decoder, buffer_view, -1, &array, nullptr),
            NANOARROW_OK);

  maybe_batch = arrow::ImportRecordBatch(&array, dummy_schema);
  ASSERT_TRUE(maybe_batch.ok());
  EXPECT_EQ(maybe_batch.ValueUnsafe()->ToString(), nulls->ToString());
  EXPECT_TRUE(maybe_batch.ValueUnsafe()->Equals(*nulls));

  schema.release(&schema);
  ArrowIpcDecoderReset(&decoder);
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, ArrowTypeParameterizedTestFixture,
    ::testing::Values(
        arrow::null(), arrow::boolean(), arrow::int8(), arrow::uint8(), arrow::int16(),
        arrow::uint16(), arrow::int32(), arrow::uint32(), arrow::int64(), arrow::uint64(),
        arrow::utf8(), arrow::float16(), arrow::float32(), arrow::float64(),
        arrow::decimal128(10, 3), arrow::decimal256(10, 3), arrow::large_utf8(),
        arrow::binary(), arrow::large_binary(), arrow::fixed_size_binary(123),
        arrow::date32(), arrow::date64(), arrow::time32(arrow::TimeUnit::SECOND),
        arrow::time32(arrow::TimeUnit::MILLI), arrow::time64(arrow::TimeUnit::MICRO),
        arrow::time64(arrow::TimeUnit::NANO), arrow::timestamp(arrow::TimeUnit::SECOND),
        arrow::timestamp(arrow::TimeUnit::MILLI),
        arrow::timestamp(arrow::TimeUnit::MICRO), arrow::timestamp(arrow::TimeUnit::NANO),
        arrow::timestamp(arrow::TimeUnit::SECOND, "UTC"),
        arrow::duration(arrow::TimeUnit::SECOND), arrow::duration(arrow::TimeUnit::MILLI),
        arrow::duration(arrow::TimeUnit::MICRO), arrow::duration(arrow::TimeUnit::NANO),
        arrow::month_interval(), arrow::day_time_interval(),
        arrow::month_day_nano_interval(),
        arrow::list(arrow::field("some_custom_name", arrow::int32())),
        arrow::large_list(arrow::field("some_custom_name", arrow::int32())),
        arrow::fixed_size_list(arrow::field("some_custom_name", arrow::int32()), 123),
        arrow::map(arrow::utf8(), arrow::int64(), false),
        arrow::map(arrow::utf8(), arrow::int64(), true),
        arrow::struct_({arrow::field("col1", arrow::int32()),
                        arrow::field("col2", arrow::utf8())}),
        // Zero-size union doesn't roundtrip through the C Data interface until
        // Arrow 11 (which is not yet available on all platforms)
        // arrow::sparse_union(FieldVector()), arrow::dense_union(FieldVector()),
        // No custom type IDs
        arrow::sparse_union({arrow::field("col1", arrow::int32()),
                             arrow::field("col2", arrow::utf8())}),
        arrow::dense_union({arrow::field("col1", arrow::int32()),
                            arrow::field("col2", arrow::utf8())}),
        // With custom type IDs
        arrow::sparse_union({arrow::field("col1", arrow::int32()),
                             arrow::field("col2", arrow::utf8())},
                            {126, 127}),
        arrow::dense_union({arrow::field("col1", arrow::int32()),
                            arrow::field("col2", arrow::utf8())},
                           {126, 127}),

        // Type with nested metadata
        arrow::list(arrow::field("some_custom_name", arrow::int32(),
                                 arrow::KeyValueMetadata::Make({"key1"}, {"value1"})))

            ));

class ArrowSchemaParameterizedTestFixture
    : public ::testing::TestWithParam<std::shared_ptr<arrow::Schema>> {
 protected:
  std::shared_ptr<arrow::Schema> arrow_schema;
};

TEST_P(ArrowSchemaParameterizedTestFixture, NanoarrowIpcArrowSchemaRoundtrip) {
  const std::shared_ptr<arrow::Schema>& arrow_schema = GetParam();
  auto maybe_serialized = arrow::ipc::SerializeSchema(*arrow_schema);
  ASSERT_TRUE(maybe_serialized.ok());

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = maybe_serialized.ValueUnsafe()->data();
  buffer_view.size_bytes = maybe_serialized.ValueOrDie()->size();

  struct ArrowIpcDecoder decoder;
  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  EXPECT_EQ(decoder.header_size_bytes, buffer_view.size_bytes);
  EXPECT_EQ(decoder.body_size_bytes, 0);

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, buffer_view, nullptr), NANOARROW_OK);
  struct ArrowSchema schema;
  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  auto maybe_schema = arrow::ImportSchema(&schema);
  ASSERT_TRUE(maybe_schema.ok());

  // Better failure message if we first check for string equality
  EXPECT_EQ(maybe_schema.ValueUnsafe()->ToString(), arrow_schema->ToString());
  EXPECT_TRUE(maybe_schema.ValueUnsafe()->Equals(arrow_schema, true));

  ArrowIpcDecoderReset(&decoder);
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcTest, ArrowSchemaParameterizedTestFixture,
    ::testing::Values(
        // Empty
        arrow::schema({}),
        // One
        arrow::schema({arrow::field("some_name", arrow::int32())}),
        // Field metadata
        arrow::schema({arrow::field(
            "some_name", arrow::int32(),
            arrow::KeyValueMetadata::Make({"key1", "key2"}, {"value1", "value2"}))}),
        // Schema metadata
        arrow::schema({}, arrow::KeyValueMetadata::Make({"key1"}, {"value1"})),
        // Non-nullable field
        arrow::schema({arrow::field("some_name", arrow::int32(), false)})));
