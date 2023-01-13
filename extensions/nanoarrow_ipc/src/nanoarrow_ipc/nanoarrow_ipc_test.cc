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
#include <gtest/gtest.h>

#include "nanoarrow_ipc.h"

using namespace arrow;

TEST(NanoarrowIpcTest, ErrorSet) {
  struct ArrowIpcError error;
  EXPECT_EQ(ArrowIpcErrorSet(&error, "there were %d foxes", 4), NANOARROW_OK);
  EXPECT_STREQ(error.message, "there were 4 foxes");
}

TEST(NanoarrowIpcTest, ErrorSetOverrun) {
  struct ArrowIpcError error;
  char big_error[2048];
  const char* a_few_chars = "abcdefg";
  for (int i = 0; i < 2047; i++) {
    big_error[i] = a_few_chars[i % strlen(a_few_chars)];
  }
  big_error[2047] = '\0';

  EXPECT_EQ(ArrowIpcErrorSet(&error, "%s", big_error), ERANGE);
  EXPECT_EQ(std::string(error.message), std::string(big_error, 1023));

  wchar_t bad_string[] = {0xFFFF, 0};
  EXPECT_EQ(ArrowIpcErrorSet(&error, "%ls", bad_string), EINVAL);
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
  struct ArrowIpcError error;

  struct ArrowIpcBufferView data;
  data.data = kSimpleSchema;
  data.size_bytes = 1;

  ArrowIpcReaderInit(&reader);

  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected data of at least 8 bytes but only 1 bytes remain");
  EXPECT_EQ(data.data, kSimpleSchema);
  EXPECT_EQ(data.size_bytes, 1);

  uint32_t eight_bad_bytes[] = {0, 0};
  data.data = reinterpret_cast<uint8_t*>(eight_bad_bytes);
  data.size_bytes = 8;
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0xFFFFFFFF at start of message but found 0x00000000");
  EXPECT_EQ(data.data, reinterpret_cast<uint8_t*>(eight_bad_bytes));
  EXPECT_EQ(data.size_bytes, 8);

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = static_cast<uint32_t>(-1);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0 <= message body size <= 0 bytes but found message body size "
               "of -1 bytes");
  EXPECT_EQ(data.data, reinterpret_cast<uint8_t*>(eight_bad_bytes));
  EXPECT_EQ(data.size_bytes, 8);

  eight_bad_bytes[1] = static_cast<uint32_t>(1);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0 <= message body size <= 0 bytes but found message body size "
               "of 1 bytes");
  EXPECT_EQ(data.data, reinterpret_cast<uint8_t*>(eight_bad_bytes));
  EXPECT_EQ(data.size_bytes, 8);

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcPeekSimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowIpcError error;

  struct ArrowIpcBufferView data;
  data.data = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);
  EXPECT_EQ(ArrowIpcReaderPeek(&reader, &data, &error), NANOARROW_OK);
  EXPECT_EQ(data.data, kSimpleSchema + sizeof(kSimpleSchema));
  EXPECT_EQ(data.size_bytes, 0);

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifySimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowIpcError error;

  struct ArrowIpcBufferView data;
  data.data = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), NANOARROW_OK);
  EXPECT_EQ(data.data, kSimpleSchema + sizeof(kSimpleSchema));
  EXPECT_EQ(data.size_bytes, 0);

  uint8_t simple_schema_invalid[280];
  memcpy(simple_schema_invalid, kSimpleSchema, sizeof(simple_schema_invalid));
  memset(simple_schema_invalid + 8, 0xFF, sizeof(simple_schema_invalid) - 8);

  data.data = simple_schema_invalid;
  data.size_bytes = sizeof(kSimpleSchema);
  EXPECT_EQ(ArrowIpcReaderVerify(&reader, &data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Message flatbuffer verification failed");
  EXPECT_EQ(data.data, simple_schema_invalid);
  EXPECT_EQ(data.size_bytes, sizeof(kSimpleSchema));

  ArrowIpcReaderReset(&reader);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleSchema) {
  struct ArrowIpcReader reader;
  struct ArrowIpcError error;

  struct ArrowIpcBufferView data;
  data.data = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcReaderInit(&reader);

  EXPECT_EQ(ArrowIpcReaderDecode(&reader, &data, &error), NANOARROW_OK);
  EXPECT_EQ(data.data, kSimpleSchema + sizeof(kSimpleSchema));
  EXPECT_EQ(data.size_bytes, 0);

  EXPECT_EQ(reader.message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);
  EXPECT_EQ(reader.endianness, NANOARROW_IPC_ENDIANNESS_LITTLE);
  EXPECT_EQ(reader.features, 0);

  ASSERT_EQ(reader.schema.n_children, 1);
  EXPECT_STREQ(reader.schema.children[0]->name, "some_col");
  EXPECT_EQ(reader.schema.children[0]->flags, ARROW_FLAG_NULLABLE);
  EXPECT_STREQ(reader.schema.children[0]->format, "i");

  ArrowIpcReaderReset(&reader);
}
