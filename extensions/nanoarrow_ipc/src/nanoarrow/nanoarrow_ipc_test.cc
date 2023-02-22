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
#include <gtest/gtest.h>

#include "nanoarrow_ipc.h"

using namespace arrow;

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

TEST(NanoarrowIpcTest, NanoarrowIpcCheckHeader) {
  struct ArrowIpcReader reader;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = 1;

  ArrowIpcReaderInit(&reader);

  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected data of at least 8 bytes but only 1 bytes remain");

  uint32_t eight_bad_bytes[] = {0, 0};
  data.data.as_uint8 = reinterpret_cast<uint8_t*>(eight_bad_bytes);
  data.size_bytes = 8;
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0xFFFFFFFF at start of message but found 0x00000000");

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = static_cast<uint32_t>(-1);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0 <= message body size <= 0 bytes but found message body size "
               "of -1 bytes");

  eight_bad_bytes[1] = static_cast<uint32_t>(1);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0 <= message body size <= 0 bytes but found message body size "
               "of 1 bytes");

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = 0;
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), ENODATA);
  EXPECT_STREQ(error.message, "End of Arrow stream");

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcPeekSimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);
  EXPECT_EQ(ArrowIpcReaderPeek(&reader, data, &error), NANOARROW_OK);
  EXPECT_EQ(reader.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(reader.body_size_bytes, 0);

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifySimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), NANOARROW_OK);
  EXPECT_EQ(reader.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(reader.body_size_bytes, 0);

  uint8_t simple_schema_invalid[280];
  memcpy(simple_schema_invalid, kSimpleSchema, sizeof(simple_schema_invalid));
  memset(simple_schema_invalid + 8, 0xFF, sizeof(simple_schema_invalid) - 8);

  data.data.as_uint8 = simple_schema_invalid;
  data.size_bytes = sizeof(kSimpleSchema);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Message flatbuffer verification failed");

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);

  EXPECT_EQ(ArrowIpcReaderDecode(&reader, data, &error), NANOARROW_OK);
  EXPECT_EQ(reader.header_size_bytes, sizeof(kSimpleSchema));
  EXPECT_EQ(reader.body_size_bytes, 0);

  EXPECT_EQ(reader.message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);
  EXPECT_EQ(reader.endianness, NANOARROW_IPC_ENDIANNESS_LITTLE);
  EXPECT_EQ(reader.features, 0);

  ASSERT_EQ(reader.schema.n_children, 1);
  EXPECT_STREQ(reader.schema.children[0]->name, "some_col");
  EXPECT_EQ(reader.schema.children[0]->flags, ARROW_FLAG_NULLABLE);
  EXPECT_STREQ(reader.schema.children[0]->format, "i");

  ArrowIpcReaderReset(&reader);
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

  struct ArrowIpcReader reader;
  ArrowIpcReaderInit(&reader);
  ASSERT_EQ(ArrowIpcReaderVerify(&reader, buffer_view, nullptr), NANOARROW_OK);
  EXPECT_EQ(reader.header_size_bytes, buffer_view.size_bytes);
  EXPECT_EQ(reader.body_size_bytes, 0);

  ASSERT_EQ(ArrowIpcReaderDecode(&reader, buffer_view, nullptr), NANOARROW_OK);
  auto maybe_schema = arrow::ImportSchema(&reader.schema);
  ASSERT_TRUE(maybe_schema.ok());

  // Better failure message if we first check for string equality
  EXPECT_EQ(maybe_schema.ValueUnsafe()->ToString(), dummy_schema->ToString());
  EXPECT_TRUE(maybe_schema.ValueUnsafe()->Equals(dummy_schema));

  ArrowIpcReaderReset(&reader);
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
                           {126, 127})));
