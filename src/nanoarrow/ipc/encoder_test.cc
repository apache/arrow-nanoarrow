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

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_ipc.hpp"

// Copied from encoder.c so we can test the internal state
extern "C" {
struct ArrowIpcEncoderPrivate {
  flatcc_builder_t builder;
  struct ArrowBuffer buffers;
  struct ArrowBuffer nodes;
};
}

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"
static_assert(sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC) == 8, "");

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderConstruction) {
  nanoarrow::ipc::UniqueEncoder encoder;

  EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  auto* p = static_cast<struct ArrowIpcEncoderPrivate*>(encoder->private_data);
  ASSERT_NE(p, nullptr);
  for (auto* b : {&p->buffers, &p->nodes}) {
    // Buffers are empty but initialized with the default allocator
    EXPECT_EQ(b->size_bytes, 0);

    auto default_allocator = ArrowBufferAllocatorDefault();
    EXPECT_EQ(memcmp(&b->allocator, &default_allocator, sizeof(b->allocator)), 0);
  }

  // Empty buffer works
  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false, buffer.get()),
      NANOARROW_OK);
  EXPECT_EQ(buffer->size_bytes, 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);
  EXPECT_EQ(buffer->size_bytes, 8);

  // Append a string (finalizing an empty buffer is an error for flatcc_builder_t)
  EXPECT_NE(flatcc_builder_create_string_str(&p->builder, "hello world"), 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false, buffer.get()),
      NANOARROW_OK);
  EXPECT_GT(buffer->size_bytes, sizeof("hello world"));

  EXPECT_NE(flatcc_builder_create_string_str(&p->builder, "hello world"), 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);
  EXPECT_GT(buffer->size_bytes, 8 + sizeof("hello world"));
  EXPECT_EQ(buffer->size_bytes % 8, 0);
}

TEST(NanoarrowIpcTest, NanoarrowIpcFooterEncoding) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueFooter footer;
  ASSERT_EQ(ArrowSchemaInitFromType(&footer->schema, NANOARROW_TYPE_STRUCT),
            NANOARROW_OK);

  nanoarrow::UniqueBuffer footer_buffer, raw_schema_buffer;
  struct ArrowError error;

  EXPECT_EQ(ArrowIpcEncoderEncodeFooter(encoder.get(), footer.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false,
                                          footer_buffer.get()),
            NANOARROW_OK);

  EXPECT_EQ(ArrowIpcEncoderEncodeSchema(encoder.get(), &footer->schema, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false,
                                          raw_schema_buffer.get()),
            NANOARROW_OK);

  EXPECT_GT(footer_buffer->size_bytes, raw_schema_buffer->size_bytes);
}
