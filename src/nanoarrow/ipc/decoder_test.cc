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

#include <cstring>
#include <thread>

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/ipc/api.h>
#include <arrow/util/key_value_metadata.h>
#endif
#include <gmock/gmock-matchers.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

// For bswap32()
#include "flatcc/portable/pendian.h"

#include "nanoarrow/nanoarrow_ipc.hpp"

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
using namespace arrow;
#endif

// Copied from decoder.c so we can test the internal state
extern "C" {
struct ArrowIpcField {
  struct ArrowArrayView* array_view;
  struct ArrowArray* array;
  int64_t buffer_offset;
};

struct ArrowIpcDecoderPrivate {
  enum ArrowIpcEndianness endianness;
  enum ArrowIpcEndianness system_endianness;
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  int64_t n_fields;
  struct ArrowIpcField* fields;
  int64_t n_buffers;
  const void* last_message;
  struct ArrowIpcFooter footer;
  struct ArrowIpcDecompressor decompressor;
};
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
alignas(8) static uint8_t kSimpleSchema[] = {
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

alignas(8) static uint8_t kSimpleRecordBatch[] = {
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

alignas(8) static uint8_t kSimpleRecordBatchCompressedZstd[] = {
    0xff, 0xff, 0xff, 0xff, 0xa0, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x18, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x20, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x1e, 0x00,
    0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x50, 0x00,
    0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00,
    0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x0c,
    0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00};

alignas(8) static uint8_t kSimpleRecordBatchCompressedLZ4[] = {
    0xff, 0xff, 0xff, 0xff, 0x98, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x18, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x28, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x1c, 0x00,
    0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x48, 0x00,
    0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x22, 0x4d, 0x18, 0x60, 0x40, 0x82, 0x0c, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00};

alignas(8) static uint8_t kSimpleRecordBatchUncompressible[] = {
    0xff, 0xff, 0xff, 0xff, 0xa0, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x00, 0x18, 0x00, 0x06, 0x00, 0x05, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x20, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x1e, 0x00,
    0x10, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x50, 0x00,
    0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x08, 0x00,
    0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00};

static uint8_t kDictionarySchema[] = {
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

static uint8_t kDictionaryBatch[] = {
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

static uint8_t kDictionaryRecordBatch[] = {
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

  // For each error, check both Verify and Decode

  ArrowIpcDecoderInit(&decoder);

  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected data of at least 8 bytes but only 1 bytes remain");

  ArrowErrorInit(&error);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected data of at least 8 bytes but only 1 bytes remain");

  uint32_t eight_bad_bytes[] = {negative_one_le * 256, 999};
  data.data.as_uint32 = eight_bad_bytes;
  data.size_bytes = sizeof(eight_bad_bytes);

#if defined(__BIG_ENDIAN__)
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(
      error.message,
      "Expected >= 16777219 bytes of remaining data but found 8 bytes in buffer");
#else
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0xFFFFFFFF at start of message but found 0xFFFFFF00");
#endif

  ArrowErrorInit(&error);

#if defined(__BIG_ENDIAN__)
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(
      error.message,
      "Expected >= 16777219 bytes of remaining data but found 8 bytes in buffer");
#else
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected 0xFFFFFFFF at start of message but found 0xFFFFFF00");
#endif

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = negative_one_le;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected message size > 0 but found message size of -1 bytes");

  ArrowErrorInit(&error);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Expected message size > 0 but found message size of -1 bytes");

  eight_bad_bytes[1] = one_le;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected >= 9 bytes of remaining data but found 8 bytes in buffer");
  ArrowErrorInit(&error);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected >= 9 bytes of remaining data but found 8 bytes in buffer");

  eight_bad_bytes[0] = 0xFFFFFFFF;
  eight_bad_bytes[1] = 0;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ENODATA);
  EXPECT_STREQ(error.message, "End of Arrow stream");
  ArrowErrorInit(&error);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), ENODATA);
  EXPECT_STREQ(error.message, "End of Arrow stream");

  uint32_t pre_continuation[] = {0, 0};
  data.data.as_uint32 = pre_continuation;
  data.size_bytes = sizeof(pre_continuation);
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ENODATA);
  EXPECT_STREQ(error.message, "End of Arrow stream");

  pre_continuation[0] = one_le << 3;
  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), ESPIPE);
  EXPECT_STREQ(error.message,
               "Expected >= 12 bytes of remaining data but found 8 bytes in buffer");

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcPeekSimpleSchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleSchema;
  data.size_bytes = sizeof(kSimpleSchema);

  ArrowIpcDecoderInit(&decoder);
  int32_t prefix_size_bytes = 0;
  EXPECT_EQ(ArrowIpcDecoderPeekHeader(&decoder, data, &prefix_size_bytes, &error),
            NANOARROW_OK);
  EXPECT_EQ(prefix_size_bytes, 8);
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

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifySimpleRecordBatch) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleRecordBatch;
  data.size_bytes = sizeof(kSimpleRecordBatch);

  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);
  EXPECT_EQ(decoder.header_size_bytes,
            sizeof(kSimpleRecordBatch) - decoder.body_size_bytes);
  EXPECT_EQ(decoder.body_size_bytes, 16);

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVerifyInvalid) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;

  uint8_t simple_schema_invalid[sizeof(kSimpleSchema)];
  struct ArrowBufferView data;
  data.data.as_uint8 = simple_schema_invalid;
  data.size_bytes = sizeof(simple_schema_invalid);

  ArrowIpcDecoderInit(&decoder);

  // Create invalid data by removing bytes one at a time and ensuring an error code and
  // a null-terminated error. After byte 265 this passes because the values being modified
  // are parts of the flatbuffer that won't cause overrun.
  for (int64_t i = 1; i < 265; i++) {
    SCOPED_TRACE(i);

    memcpy(simple_schema_invalid, kSimpleSchema, i);
    memcpy(simple_schema_invalid + i, kSimpleSchema + (i + 1),
           (sizeof(simple_schema_invalid) - i - 1));

    ArrowErrorInit(&error);
    ASSERT_NE(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), NANOARROW_OK);
    ASSERT_GT(strlen(error.message), 0);
  }

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

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}

void TestDecodeInt32Batch(const uint8_t* batch, size_t batch_len,
                          const std::vector<int32_t> values) {
  nanoarrow::ipc::UniqueDecoder decoder;
  nanoarrow::UniqueSchema schema;

  struct ArrowError error;
  struct ArrowArrayView* array_view;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  struct ArrowBufferView data;
  data.data.as_uint8 = batch;
  data.size_bytes = static_cast<int64_t>(batch_len);

  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), schema.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcDecoderSetEndianness(decoder.get(), NANOARROW_IPC_ENDIANNESS_LITTLE),
            NANOARROW_OK);

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), data, &error), NANOARROW_OK);
  struct ArrowBufferView body;
  body.data.as_uint8 = batch + decoder->header_size_bytes;
  body.size_bytes = decoder->body_size_bytes;

  ASSERT_EQ(ArrowIpcDecoderDecodeArrayView(decoder.get(), body, 0, &array_view, &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(array_view->length, values.size());
  int64_t index = 0;
  for (const auto value : values) {
    SCOPED_TRACE(std::string("Array index ") + std::to_string(index));
    EXPECT_EQ(ArrowArrayViewGetIntUnsafe(array_view, index), value);
    index++;
  }
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleRecordBatch) {
  ASSERT_NO_FATAL_FAILURE(
      TestDecodeInt32Batch(kSimpleRecordBatch, sizeof(kSimpleRecordBatch), {1, 2, 3}));
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeCompressedRecordBatchZstd) {
  if (ArrowIpcGetZstdDecompressionFunction() == nullptr) {
    EXPECT_FATAL_FAILURE(
        TestDecodeInt32Batch(kSimpleRecordBatchCompressedZstd,
                             sizeof(kSimpleRecordBatchCompressedZstd), {0, 1, 2}),
        "Compression type with value 2 not supported by this build of nanoarrow");
  } else {
    ASSERT_NO_FATAL_FAILURE(TestDecodeInt32Batch(kSimpleRecordBatchCompressedZstd,
                                                 sizeof(kSimpleRecordBatchCompressedZstd),
                                                 {0, 1, 2}));
  }
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeCompressedRecordBatchLZ4) {
  if (ArrowIpcGetLZ4DecompressionFunction() == nullptr) {
    EXPECT_FATAL_FAILURE(
        TestDecodeInt32Batch(kSimpleRecordBatchCompressedLZ4,
                             sizeof(kSimpleRecordBatchCompressedLZ4), {0, 1, 2}),
        "Compression type with value 1 not supported by this build of nanoarrow");
  } else {
    ASSERT_NO_FATAL_FAILURE(TestDecodeInt32Batch(kSimpleRecordBatchCompressedLZ4,
                                                 sizeof(kSimpleRecordBatchCompressedLZ4),
                                                 {0, 1, 2}));
  }
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeUncompressibleRecordBatch) {
  ASSERT_NO_FATAL_FAILURE(TestDecodeInt32Batch(kSimpleRecordBatchUncompressible,
                                               sizeof(kSimpleRecordBatchUncompressible),
                                               {0, 1, 2}));
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleRecordBatchErrors) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;
  struct ArrowArray array;

  // Data buffer content of the hard-coded record batch message
  uint8_t one_two_three_le[] = {0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
                                0x00, 0x00, 0x03, 0x00, 0x00, 0x00};

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
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, data, 0, nullptr,
                                       NANOARROW_VALIDATION_LEVEL_FULL, &error),
            EINVAL);
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
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, -1, &array,
                                       NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  ASSERT_EQ(array.n_children, 1);
  ASSERT_EQ(array.children[0]->n_buffers, 2);
  ASSERT_EQ(array.children[0]->length, 3);
  EXPECT_EQ(array.children[0]->null_count, 0);
  EXPECT_EQ(
      memcmp(array.children[0]->buffers[1], one_two_three_le, sizeof(one_two_three_le)),
      0);

  ArrowArrayRelease(&array);

  // Check field extract
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array,
                                       NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
            NANOARROW_OK);
  ASSERT_EQ(array.n_buffers, 2);
  ASSERT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  EXPECT_EQ(memcmp(array.buffers[1], one_two_three_le, sizeof(one_two_three_le)), 0);

  ArrowArrayRelease(&array);

  // Field extract should fail if body is too small
  decoder.body_size_bytes = 0;
  EXPECT_EQ(ArrowIpcDecoderDecodeArray(&decoder, body, 0, &array,
                                       NANOARROW_VALIDATION_LEVEL_FULL, &error),
            EINVAL);
  EXPECT_STREQ(error.message, "Buffer requires body offsets [0..12) but body has size 0");

  // Should error if the number of buffers or field nodes doesn't match
  // (different numbers because we count the root struct and the message does not)
  decoder_private->n_buffers = 1;
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Expected 0 buffers in message but found 2");

  decoder_private->n_fields = 1;
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), EINVAL);
  EXPECT_STREQ(error.message, "Expected 0 field nodes in message but found 1");

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeDictionarySchema) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;

  struct ArrowBufferView data;
  data.data.as_uint8 = kDictionarySchema;
  data.size_bytes = sizeof(kDictionarySchema);

  ASSERT_EQ(ArrowIpcDecoderInit(&decoder), NANOARROW_OK);

  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), NANOARROW_OK);
  ASSERT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);

  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(&decoder, &schema, &error), NANOARROW_OK);
  ASSERT_EQ(schema.n_children, 1);
  EXPECT_STREQ(schema.children[0]->name, "some_col");
  EXPECT_EQ(schema.children[0]->flags, ARROW_FLAG_NULLABLE);
  EXPECT_STREQ(schema.children[0]->format, "c");

  ASSERT_NE(schema.children[0]->dictionary, nullptr);
  EXPECT_STREQ(schema.children[0]->dictionary->format, "u");

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeDictionaryBatch) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;

  struct ArrowBufferView data;
  data.data.as_uint8 = kDictionaryBatch;
  data.size_bytes = sizeof(kDictionaryBatch);

  ASSERT_EQ(ArrowIpcDecoderInit(&decoder), NANOARROW_OK);

  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), NANOARROW_OK);
  ASSERT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH);

  ASSERT_NE(decoder.dictionary, nullptr);
  EXPECT_EQ(decoder.dictionary->id, 0);
  EXPECT_FALSE(decoder.dictionary->is_delta);

  // TODO: Access RecordBatch content

  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeDictionaryRecordBatch) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;

  struct ArrowBufferView data;
  data.data.as_uint8 = kDictionaryRecordBatch;
  data.size_bytes = sizeof(kDictionaryRecordBatch);

  ASSERT_EQ(ArrowIpcDecoderInit(&decoder), NANOARROW_OK);

  EXPECT_EQ(ArrowIpcDecoderVerifyHeader(&decoder, data, &error), NANOARROW_OK);
  ASSERT_EQ(decoder.message_type, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);

  // TODO: Decode RecordBatch content populating dictionary array member

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

  // Make sure we can re-set a schema too
  EXPECT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(decoder_private->n_fields, 2);
  EXPECT_EQ(decoder_private->n_buffers, 3);

  ArrowSchemaRelease(&schema);
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

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcSetDecompressor) {
  struct ArrowIpcDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(&decoder), NANOARROW_OK);

  struct ArrowIpcDecompressor decompressor;
  ASSERT_EQ(ArrowIpcSerialDecompressor(&decompressor), NANOARROW_OK);
  EXPECT_NE(decompressor.release, nullptr);

  ASSERT_EQ(ArrowIpcDecoderSetDecompressor(&decoder, &decompressor), NANOARROW_OK);
  ASSERT_EQ(decompressor.release, nullptr);

  ArrowIpcDecoderReset(&decoder);
}

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
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
#endif

std::string ArrowSchemaMetadataToString(const char* metadata) {
  struct ArrowMetadataReader reader;
  auto st = ArrowMetadataReaderInit(&reader, metadata);
  EXPECT_EQ(st, NANOARROW_OK);

  bool comma = false;
  std::string out;
  while (reader.remaining_keys > 0) {
    struct ArrowStringView key, value;
    auto st = ArrowMetadataReaderRead(&reader, &key, &value);
    EXPECT_EQ(st, NANOARROW_OK);
    if (comma) {
      out += ", ";
    }
    comma = true;

    out.append(key.data, key.size_bytes);
    out += "=";
    out.append(value.data, value.size_bytes);
  }
  return out;
}

std::string ArrowSchemaToString(const struct ArrowSchema* schema) {
  int64_t n = ArrowSchemaToString(schema, nullptr, 0, /*recursive=*/false);
  std::vector<char> out_vec(n, '\0');
  ArrowSchemaToString(schema, out_vec.data(), n, /*recursive=*/false);
  std::string out(out_vec.data(), out_vec.size());

  std::string metadata = ArrowSchemaMetadataToString(schema->metadata);
  if (!metadata.empty()) {
    out += "{" + metadata + "}";
  }

  bool comma = false;
  if (schema->format[0] == '+') {
    out += "<";
    for (int64_t i = 0; i < schema->n_children; ++i) {
      if (comma) {
        out += ", ";
      }
      comma = true;

      auto* child = schema->children[i];
      if (child && child->name[0] != '\0') {
        out += child->name;
        out += ": ";
      }
      out += ArrowSchemaToString(schema->children[i]);
    }
    out += ">";
  }

  return out;
}

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
TEST_P(ArrowTypeParameterizedTestFixture, NanoarrowIpcNanoarrowTypeRoundtrip) {
  nanoarrow::UniqueSchema schema;
  ASSERT_TRUE(
      arrow::ExportSchema(arrow::Schema({arrow::field("", GetParam())}), schema.get())
          .ok());

  nanoarrow::ipc::UniqueEncoder encoder;
  EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  struct ArrowError error;
  EXPECT_EQ(ArrowIpcEncoderEncodeSchema(encoder.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = buffer->data;
  buffer_view.size_bytes = buffer->size_bytes;

  nanoarrow::ipc::UniqueDecoder decoder;
  ArrowIpcDecoderInit(decoder.get());
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(decoder.get(), buffer_view, nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), buffer_view, nullptr),
            NANOARROW_OK);

  nanoarrow::UniqueSchema roundtripped;
  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(decoder.get(), roundtripped.get(), nullptr),
            NANOARROW_OK);

  EXPECT_EQ(ArrowSchemaToString(roundtripped.get()), ArrowSchemaToString(schema.get()));
}
#endif

TEST(NanoarrowIpcTest, NanoarrowIpcDecodeSimpleRecordBatchFromShared) {
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  struct ArrowSchema schema;
  struct ArrowArray array;

  // Data buffer content of the hard-coded record batch message
  uint8_t one_two_three_le[] = {0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
                                0x00, 0x00, 0x03, 0x00, 0x00, 0x00};

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeStruct(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleRecordBatch;
  data.size_bytes = sizeof(kSimpleRecordBatch);

  ArrowIpcDecoderInit(&decoder);

  ASSERT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, &error), NANOARROW_OK);

  struct ArrowBuffer body;
  ArrowBufferInit(&body);
  ASSERT_EQ(ArrowBufferAppend(&body, kSimpleRecordBatch + decoder.header_size_bytes,
                              decoder.body_size_bytes),
            NANOARROW_OK);

  struct ArrowIpcSharedBuffer shared;
  ASSERT_EQ(ArrowIpcSharedBufferInit(&shared, &body), NANOARROW_OK);

  // Check full struct extract
  EXPECT_EQ(ArrowIpcDecoderDecodeArrayFromShared(
                &decoder, &shared, -1, &array, NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  ASSERT_EQ(array.n_children, 1);
  ASSERT_EQ(array.children[0]->n_buffers, 2);
  ASSERT_EQ(array.children[0]->length, 3);
  EXPECT_EQ(array.children[0]->null_count, 0);
  EXPECT_EQ(
      memcmp(array.children[0]->buffers[1], one_two_three_le, sizeof(one_two_three_le)),
      0);

  ArrowArrayRelease(&array);

  // Check field extract
  EXPECT_EQ(ArrowIpcDecoderDecodeArrayFromShared(
                &decoder, &shared, 0, &array, NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
            NANOARROW_OK);

  // Release the original shared (forthcoming array buffers should still be valid)
  ArrowIpcSharedBufferReset(&shared);

  ASSERT_EQ(array.n_buffers, 2);
  ASSERT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 0);
  EXPECT_EQ(memcmp(array.buffers[1], one_two_three_le, sizeof(one_two_three_le)), 0);

  ArrowArrayRelease(&array);
  ArrowSchemaRelease(&schema);
  ArrowBufferReset(&body);
  ArrowIpcDecoderReset(&decoder);
}

TEST(NanoarrowIpcTest, NanoarrowIpcSharedBufferThreadSafeDecode) {
  if (!ArrowIpcSharedBufferIsThreadSafe()) {
    GTEST_SKIP() << "ArrowIpcSharedBufferIsThreadSafe() returned false";
  }

  struct ArrowIpcDecoder decoder;
  struct ArrowSchema schema;

  // Data buffer content of the hard-coded record batch message
  uint8_t one_two_three_le[] = {0x01, 0x00, 0x00, 0x00, 0x02, 0x00,
                                0x00, 0x00, 0x03, 0x00, 0x00, 0x00};

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeStruct(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  struct ArrowBufferView data;
  data.data.as_uint8 = kSimpleRecordBatch;
  data.size_bytes = sizeof(kSimpleRecordBatch);

  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, nullptr), NANOARROW_OK);

  struct ArrowBuffer body;
  ArrowBufferInit(&body);
  ASSERT_EQ(ArrowBufferAppend(&body, kSimpleRecordBatch + decoder.header_size_bytes,
                              decoder.body_size_bytes),
            NANOARROW_OK);

  struct ArrowIpcSharedBuffer shared;
  ASSERT_EQ(ArrowIpcSharedBufferInit(&shared, &body), NANOARROW_OK);

  struct ArrowArray arrays[10];
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(
        ArrowIpcDecoderDecodeArrayFromShared(&decoder, &shared, -1, arrays + i,
                                             NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
        NANOARROW_OK);
  }

  // Clean up
  ArrowIpcSharedBufferReset(&shared);
  ArrowIpcDecoderReset(&decoder);
  ArrowSchemaRelease(&schema);

  // Access the data and release from another thread
  std::thread threads[10];
  for (int i = 0; i < 10; i++) {
    threads[i] = std::thread([&arrays, i, &one_two_three_le] {
      auto result = memcmp(arrays[i].children[0]->buffers[1], one_two_three_le,
                           sizeof(one_two_three_le));
      // discard result to silence -Wunused-value
      NANOARROW_UNUSED(result);
      ArrowArrayRelease(arrays + i);
    });
  }

  for (int i = 0; i < 10; i++) {
    threads[i].join();
  }

  // We will get a (occasional) memory leak if the atomic counter does not work
}

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
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
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(&decoder, buffer_view, -1, &array,
                                       NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
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
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(&decoder, buffer_view, -1, &array,
                                       NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
            NANOARROW_OK);

  maybe_batch = arrow::ImportRecordBatch(&array, dummy_schema);
  ASSERT_TRUE(maybe_batch.ok());
  EXPECT_EQ(maybe_batch.ValueUnsafe()->ToString(), nulls->ToString());
  EXPECT_TRUE(maybe_batch.ValueUnsafe()->Equals(*nulls));

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}
#endif

void AssertArrayViewIdentical(const struct ArrowArrayView* actual,
                              const struct ArrowArrayView* expected) {
  NANOARROW_DCHECK(actual->dictionary == nullptr);
  NANOARROW_DCHECK(expected->dictionary == nullptr);

  ASSERT_EQ(actual->storage_type, expected->storage_type);
  ASSERT_EQ(actual->offset, expected->offset);
  ASSERT_EQ(actual->length, expected->length);
  for (int i = 0; i < 3; i++) {
    auto a_buf = actual->buffer_views[i];
    auto e_buf = expected->buffer_views[i];
    ASSERT_EQ(a_buf.size_bytes, e_buf.size_bytes);
    if (a_buf.size_bytes != 0) {
      ASSERT_EQ(memcmp(a_buf.data.data, e_buf.data.data, a_buf.size_bytes), 0);
    }
  }

  ASSERT_EQ(actual->n_children, expected->n_children);
  for (int i = 0; i < actual->n_children; i++) {
    AssertArrayViewIdentical(actual->children[i], expected->children[i]);
  }
}

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
TEST_P(ArrowTypeParameterizedTestFixture, NanoarrowIpcNanoarrowArrayRoundtrip) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  ASSERT_TRUE(
      arrow::ExportSchema(arrow::Schema({arrow::field("", GetParam())}), schema.get())
          .ok());

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK);

  // now make one empty struct array with this schema and another with all zeroes
  nanoarrow::UniqueArray empty_array, zero_array;
  for (auto* array : {empty_array.get(), zero_array.get()}) {
    ASSERT_EQ(ArrowArrayInitFromSchema(array, schema.get(), nullptr), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
    if (array == zero_array.get()) {
      ASSERT_EQ(ArrowArrayAppendEmpty(array, 5), NANOARROW_OK);
    }
    ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array, &error), NANOARROW_OK)
        << error.message;

    nanoarrow::ipc::UniqueEncoder encoder;
    EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

    nanoarrow::UniqueBuffer buffer, body_buffer;
    EXPECT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), array_view.get(),
                                                     body_buffer.get(), &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(
        ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
        NANOARROW_OK);

    nanoarrow::ipc::UniqueDecoder decoder;
    ArrowIpcDecoderInit(decoder.get());
    EXPECT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), schema.get(), &error), NANOARROW_OK)
        << error.message;
    EXPECT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(),
                                          {{buffer->data}, buffer->size_bytes}, &error),
              NANOARROW_OK)
        << error.message;

    struct ArrowArrayView* roundtripped;
    ASSERT_EQ(ArrowIpcDecoderDecodeArrayView(
                  decoder.get(), {{body_buffer->data}, body_buffer->size_bytes}, -1,
                  &roundtripped, nullptr),
              NANOARROW_OK);

    AssertArrayViewIdentical(roundtripped, array_view.get());
  }
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
        arrow::month_day_nano_interval(), arrow::list(arrow::int32()),
        arrow::list(arrow::field("", arrow::int32())),
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
  EXPECT_EQ(maybe_schema.ValueUnsafe()->ToString(/*show_metadata=*/true),
            arrow_schema->ToString(/*show_metadata=*/true));
  EXPECT_TRUE(maybe_schema.ValueUnsafe()->Equals(arrow_schema, true));

  ArrowIpcDecoderReset(&decoder);
}

TEST_P(ArrowSchemaParameterizedTestFixture, NanoarrowIpcNanoarrowSchemaRoundtrip) {
  const std::shared_ptr<arrow::Schema>& arrow_schema = GetParam();

  nanoarrow::UniqueSchema schema;
  ASSERT_TRUE(arrow::ExportSchema(*arrow_schema, schema.get()).ok());

  nanoarrow::ipc::UniqueEncoder encoder;
  EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  struct ArrowError error;
  EXPECT_EQ(ArrowIpcEncoderEncodeSchema(encoder.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = buffer->data;
  buffer_view.size_bytes = buffer->size_bytes;

  nanoarrow::ipc::UniqueDecoder decoder;
  ArrowIpcDecoderInit(decoder.get());
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(decoder.get(), buffer_view, nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), buffer_view, nullptr),
            NANOARROW_OK);

  nanoarrow::UniqueSchema roundtripped;
  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(decoder.get(), roundtripped.get(), nullptr),
            NANOARROW_OK);

  EXPECT_EQ(ArrowSchemaToString(roundtripped.get()), ArrowSchemaToString(schema.get()));
}

TEST_P(ArrowSchemaParameterizedTestFixture, NanoarrowIpcNanoarrowFooterRoundtrip) {
  using namespace nanoarrow::literals;
  const std::shared_ptr<arrow::Schema>& arrow_schema = GetParam();

  nanoarrow::ipc::UniqueFooter footer;
  ASSERT_TRUE(arrow::ExportSchema(*arrow_schema, &footer->schema).ok());

  struct ArrowIpcFileBlock dummy_block = {1, 2, 3};
  EXPECT_EQ(
      ArrowBufferAppend(&footer->record_batch_blocks, &dummy_block, sizeof(dummy_block)),
      NANOARROW_OK);

  nanoarrow::ipc::UniqueEncoder encoder;
  EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  struct ArrowError error;
  EXPECT_EQ(ArrowIpcEncoderEncodeFooter(encoder.get(), footer.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false, buffer.get()),
      NANOARROW_OK);

#ifdef __BIG_ENDIAN__
  uint32_t footer_size_le = bswap32(static_cast<uint32_t>(buffer->size_bytes));
  EXPECT_EQ(ArrowBufferAppendInt32(buffer.get(), footer_size_le), NANOARROW_OK);
#else
  EXPECT_EQ(ArrowBufferAppendInt32(buffer.get(), buffer->size_bytes), NANOARROW_OK);
#endif
  EXPECT_EQ(ArrowBufferAppendStringView(buffer.get(), "ARROW1"_asv), NANOARROW_OK);

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = buffer->data;
  buffer_view.size_bytes = buffer->size_bytes;

  nanoarrow::ipc::UniqueDecoder decoder;
  ArrowIpcDecoderInit(decoder.get());
  ASSERT_EQ(ArrowIpcDecoderVerifyFooter(decoder.get(), buffer_view, &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcDecoderDecodeFooter(decoder.get(), buffer_view, &error), NANOARROW_OK)
      << error.message;

  EXPECT_EQ(ArrowSchemaToString(&decoder->footer->schema),
            ArrowSchemaToString(&footer->schema));
  EXPECT_EQ(decoder->footer->record_batch_blocks.size_bytes, sizeof(dummy_block));

  struct ArrowIpcFileBlock roundtripped_block;
  memcpy(&roundtripped_block, decoder->footer->record_batch_blocks.data,
         sizeof(roundtripped_block));

  EXPECT_EQ(roundtripped_block.offset, dummy_block.offset);
  EXPECT_EQ(roundtripped_block.metadata_length, dummy_block.metadata_length);
  EXPECT_EQ(roundtripped_block.body_length, dummy_block.body_length);
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

class ArrowTypeIdParameterizedTestFixture
    : public ::testing::TestWithParam<enum ArrowType> {
 protected:
  enum ArrowType data_type;
};

TEST_P(ArrowTypeIdParameterizedTestFixture, NanoarrowIpcDecodeSwapEndian) {
  enum ArrowType data_type = GetParam();
  int64_t n_elements_test = 10;

  // Make a data buffer long enough for 10 Decimal256s with a pattern
  // where an endian swap isn't silently the same value (e.g., 0s)
  uint8_t data_buffer[32 * 10];
  for (size_t i = 0; i < sizeof(data_buffer); i++) {
    data_buffer[i] = i % 256;
  }

  int bit_width;
  std::shared_ptr<arrow::DataType> arrow_data_type;
  switch (data_type) {
    case NANOARROW_TYPE_BOOL:
      bit_width = 1;
      arrow_data_type = arrow::boolean();
      break;
    case NANOARROW_TYPE_INT8:
      bit_width = 8;
      arrow_data_type = arrow::int8();
      break;
    case NANOARROW_TYPE_INT16:
      bit_width = 16;
      arrow_data_type = arrow::int16();
      break;
    case NANOARROW_TYPE_INT32:
      bit_width = 32;
      arrow_data_type = arrow::int32();
      break;
    case NANOARROW_TYPE_INT64:
      bit_width = 64;
      arrow_data_type = arrow::int64();
      break;
    case NANOARROW_TYPE_DECIMAL128:
      arrow_data_type = arrow::decimal128(10, 3);
      bit_width = 128;
      break;
    case NANOARROW_TYPE_DECIMAL256:
      bit_width = 256;
      arrow_data_type = arrow::decimal256(10, 3);
      break;
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      bit_width = 128;
      arrow_data_type = arrow::month_day_nano_interval();
      break;
    default:
      GTEST_FAIL() << "Type not supported for test";
  }

  // "Manually" swap the endians
  uint8_t data_buffer_swapped[32 * 10];
  int64_t n_elements = sizeof(data_buffer) * 8 / bit_width;
  if (bit_width > 8 && data_type != NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO) {
    int byte_width = bit_width / 8;
    for (int64_t i = 0; i < n_elements; i++) {
      uint8_t* src = data_buffer + (i * byte_width);
      uint8_t* dst = data_buffer_swapped + (i * byte_width);
      for (int j = 0; j < byte_width; j++) {
        dst[j] = src[byte_width - j - 1];
      }
    }
  } else if (data_type == NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO) {
    for (int64_t i = 0; i < n_elements; i++) {
      uint8_t* src = data_buffer + (i * 16);
      uint8_t* dst = data_buffer_swapped + (i * 16);

      for (int j = 0; j < 4; j++) {
        dst[j] = src[4 - j - 1];
      }
      src += 4;
      dst += 4;

      for (int j = 0; j < 4; j++) {
        dst[j] = src[4 - j - 1];
      }
      src += 4;
      dst += 4;

      for (int j = 0; j < 8; j++) {
        dst[j] = src[8 - j - 1];
      }
    }
  } else {
    memcpy(data_buffer_swapped, data_buffer, sizeof(data_buffer));
  }

  // Make an array wrapping the swapped buffer
  auto empty = std::make_shared<arrow::Buffer>(nullptr, 0);
  auto buffer =
      std::make_shared<arrow::Buffer>(data_buffer_swapped, sizeof(data_buffer_swapped));
  arrow::BufferVector buffers = {empty, buffer};
  auto array_data =
      std::make_shared<arrow::ArrayData>(arrow_data_type, n_elements_test, buffers, 0, 0);
  auto array = arrow::MakeArray(array_data);

  // Make a RecordBatch
  auto arrow_schema = arrow::schema({arrow::field("col1", arrow_data_type)});
  auto arrow_record_batch =
      arrow::RecordBatch::Make(arrow_schema, n_elements_test, {array});

  // Serialize it
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  auto maybe_serialized = arrow::ipc::SerializeRecordBatch(*arrow_record_batch, options);
  if (!maybe_serialized.ok()) {
    GTEST_FAIL() << maybe_serialized.status();
  }
  auto serialized = *maybe_serialized;

  struct ArrowSchema schema;
  if (!arrow::ExportSchema(*arrow_schema, &schema).ok()) {
    GTEST_FAIL() << "schema export failed";
  }

  struct ArrowBufferView data;
  data.data.as_uint8 = serialized->data();
  data.size_bytes = serialized->size();

  struct ArrowIpcDecoder decoder;
  ArrowIpcDecoderInit(&decoder);
  ASSERT_EQ(ArrowIpcDecoderSetSchema(&decoder, &schema, nullptr), NANOARROW_OK);

#ifdef __BIG_ENDIAN__
  ASSERT_EQ(ArrowIpcDecoderSetEndianness(&decoder, NANOARROW_IPC_ENDIANNESS_LITTLE),
            NANOARROW_OK);
#else
  ASSERT_EQ(ArrowIpcDecoderSetEndianness(&decoder, NANOARROW_IPC_ENDIANNESS_BIG),
            NANOARROW_OK);
#endif

  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(&decoder, data, nullptr), NANOARROW_OK);
  data.data.as_uint8 += decoder.header_size_bytes;
  data.size_bytes -= decoder.header_size_bytes;

  struct ArrowArrayView* array_view;
  ASSERT_EQ(ArrowIpcDecoderDecodeArrayView(&decoder, data, 0, &array_view, nullptr),
            NANOARROW_OK);
  ASSERT_EQ(array_view->storage_type, data_type);

  // Check buffer equality with our initial buffer
  EXPECT_EQ(memcmp(array_view->buffer_views[1].data.data, data_buffer,
                   array_view->buffer_views[1].size_bytes),
            0);

  ArrowSchemaRelease(&schema);
  ArrowIpcDecoderReset(&decoder);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowIpcTest, ArrowTypeIdParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_BOOL, NANOARROW_TYPE_INT8,
                                           NANOARROW_TYPE_INT16, NANOARROW_TYPE_INT32,
                                           NANOARROW_TYPE_INT64,
                                           NANOARROW_TYPE_DECIMAL128,
                                           NANOARROW_TYPE_DECIMAL256,
                                           NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO));
#endif

TEST(NanoarrowIpcTest, NanoarrowIpcFooterDecodingErrors) {
  struct ArrowError error;

  nanoarrow::ipc::UniqueDecoder decoder;
  ArrowIpcDecoderInit(decoder.get());

  // not enough data to get the size+magic
  EXPECT_EQ(ArrowIpcDecoderPeekFooter(decoder.get(), {{nullptr}, 3}, &error), ESPIPE)
      << error.message;

  // doesn't end with magic
  EXPECT_EQ(ArrowIpcDecoderPeekFooter(decoder.get(), {{"\0\0\0\0blargh"}, 10}, &error),
            EINVAL)
      << error.message;

  // negative size
  EXPECT_EQ(ArrowIpcDecoderPeekFooter(decoder.get(),
                                      {{"\xFF\xFF\xFF\xFF"
                                        "ARROW1"},
                                       10},
                                      &error),
            EINVAL)
      << error.message;

  // PeekFooter doesn't check for available data
  EXPECT_EQ(
      ArrowIpcDecoderPeekFooter(decoder.get(), {{"\xFF\xFF\0\0ARROW1"}, 10}, &error),
      NANOARROW_OK)
      << error.message;
  EXPECT_EQ(decoder->header_size_bytes, 0xFFFF);

  decoder->header_size_bytes = -1;

  // VerifyFooter *does* check for enough available data
  EXPECT_EQ(
      ArrowIpcDecoderVerifyFooter(decoder.get(), {{"\xFF\xFF\0\0ARROW1"}, 10}, &error),
      ESPIPE)
      << error.message;
  EXPECT_EQ(decoder->header_size_bytes, 0xFFFF);
}
