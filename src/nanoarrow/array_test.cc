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

#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/testing/gtest_util.h>

#include "nanoarrow/nanoarrow.h"

using namespace arrow;

TEST(ArrayTest, ArrayTestInit) {
  struct ArrowArray array;

  EXPECT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_UNINITIALIZED), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 0);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 1);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 2);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 3);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_DATE64), EINVAL);
}

TEST(ArrayTest, ArrayTestAllocateChildren) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 0), NANOARROW_OK);
  EXPECT_EQ(array.n_children, 0);
  array.release(&array);

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(&array, std::numeric_limits<int64_t>::max()),
            ENOMEM);
  array.release(&array);

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 2), NANOARROW_OK);
  EXPECT_EQ(array.n_children, 2);
  ASSERT_NE(array.children, nullptr);
  ASSERT_NE(array.children[0], nullptr);
  ASSERT_NE(array.children[1], nullptr);

  ASSERT_EQ(ArrowArrayInit(array.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInit(array.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 0), EINVAL);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestAllocateDictionary) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateDictionary(&array), NANOARROW_OK);
  ASSERT_NE(array.dictionary, nullptr);

  ASSERT_EQ(ArrowArrayInit(array.dictionary, NANOARROW_TYPE_STRING), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAllocateDictionary(&array), EINVAL);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestInitFromSchema) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 2), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayInitFromSchema(&array, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(array.n_children, 2);
  EXPECT_EQ(array.children[0]->n_buffers, 2);
  EXPECT_EQ(array.children[1]->n_buffers, 3);

  array.release(&array);
  schema.release(&schema);
}

TEST(ArrayTest, ArrayTestSetBitmap) {
  struct ArrowBitmap bitmap;
  ArrowBitmapInit(&bitmap);
  ArrowBitmapAppend(&bitmap, true, 9);

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ArrowArraySetValidityBitmap(&array, &bitmap);
  EXPECT_EQ(bitmap.buffer.data, nullptr);
  const uint8_t* bitmap_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  EXPECT_EQ(bitmap_buffer[0], 0xff);
  EXPECT_EQ(bitmap_buffer[1], 0x01);
  EXPECT_EQ(ArrowArrayValidityBitmap(&array)->buffer.data, array.buffers[0]);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestSetBuffer) {
  // the array ["a", null, "bc", null, "def", null, "ghij"]
  uint8_t validity_bitmap[] = {0x55};
  int32_t offsets[] = {0, 1, 1, 3, 3, 6, 6, 10, 10};
  const char* data = "abcdefghij";

  struct ArrowBuffer buffer0, buffer1, buffer2;
  ArrowBufferInit(&buffer0);
  ArrowBufferAppend(&buffer0, validity_bitmap, 1);
  ArrowBufferInit(&buffer1);
  ArrowBufferAppend(&buffer1, offsets, 9 * sizeof(int32_t));
  ArrowBufferInit(&buffer2);
  ArrowBufferAppend(&buffer2, data, strlen(data));

  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowArraySetBuffer(&array, 0, &buffer0), NANOARROW_OK);
  EXPECT_EQ(ArrowArraySetBuffer(&array, 1, &buffer1), NANOARROW_OK);
  EXPECT_EQ(ArrowArraySetBuffer(&array, 2, &buffer2), NANOARROW_OK);

  EXPECT_EQ(memcmp(array.buffers[0], validity_bitmap, 1), 0);
  EXPECT_EQ(memcmp(array.buffers[1], offsets, 8 * sizeof(int32_t)), 0);
  EXPECT_EQ(memcmp(array.buffers[2], data, 10), 0);

  EXPECT_EQ(ArrowArrayBuffer(&array, 0)->data, array.buffers[0]);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->data, array.buffers[1]);
  EXPECT_EQ(ArrowArrayBuffer(&array, 2)->data, array.buffers[2]);

  // try to set a buffer that isn't, 0, 1, or 2
  EXPECT_EQ(ArrowArraySetBuffer(&array, 3, &buffer0), EINVAL);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestBuildByBuffer) {
  // the array ["a", null, "bc", null, "def", null, "ghij"]
  uint8_t validity_bitmap[] = {0x55};
  int8_t validity_array[] = {1, 0, 1, 0, 1, 0, 1};
  int32_t offsets[] = {0, 1, 1, 3, 3, 6, 6, 10, 10};
  const char* data = "abcdefghij";

  struct ArrowArray array;
  struct ArrowError error;
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);

  ASSERT_EQ(ArrowBitmapReserve(ArrowArrayValidityBitmap(&array), 100), NANOARROW_OK);
  ArrowBitmapAppendInt8Unsafe(ArrowArrayValidityBitmap(&array), validity_array, 7);

  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 1), 100), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 1), offsets, 8 * sizeof(int32_t));

  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 2), 100), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 2), data, 10);

  array.length = 7;
  EXPECT_EQ(ArrowArrayShrinkToFit(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), NANOARROW_OK);

  EXPECT_EQ(memcmp(array.buffers[0], validity_bitmap, 1), 0);
  EXPECT_EQ(memcmp(array.buffers[1], offsets, 8 * sizeof(int32_t)), 0);
  EXPECT_EQ(memcmp(array.buffers[2], data, 10), 0);

  EXPECT_EQ(ArrowArrayBuffer(&array, 0)->data, array.buffers[0]);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->data, array.buffers[1]);
  EXPECT_EQ(ArrowArrayBuffer(&array, 2)->data, array.buffers[2]);

  EXPECT_EQ(ArrowArrayBuffer(&array, 0)->size_bytes, 1);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->size_bytes, 8 * sizeof(int32_t));
  EXPECT_EQ(ArrowArrayBuffer(&array, 2)->size_bytes, 10);

  array.length = 8;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected buffer 1 to size >= 36 bytes but found buffer with 32 bytes");

  array.length = 7;
  int32_t* offsets_buffer = reinterpret_cast<int32_t*>(ArrowArrayBuffer(&array, 1)->data);
  offsets_buffer[7] = offsets_buffer[7] + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected buffer 2 to size >= 11 bytes but found buffer with 10 bytes");

  array.release(&array);
}

TEST(ArrayTest, ArrayTestAppendToNullArray) {
  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_NA), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 2);
  EXPECT_EQ(array.null_count, 2);

  auto arrow_array = ImportArray(&array, null());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(null(), "[null, null]")));

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_NA), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 0), EINVAL);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 0), EINVAL);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 0), EINVAL);
  struct ArrowBufferView buffer_view;
  buffer_view.data.data = nullptr;
  buffer_view.n_bytes = 0;
  EXPECT_EQ(ArrowArrayAppendBytes(&array, buffer_view), EINVAL);
  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("")), EINVAL);
  array.release(&array);
}

TEST(ArrayTest, ArrayTestAppendToInt64Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT64), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const int64_t*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);

  auto arrow_array = ImportArray(&array, int64());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(
      arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(int64(), "[1, null, null, 3]")));
}

TEST(ArrayTest, ArrayTestAppendToInt32Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, std::numeric_limits<int64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 1);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const int32_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 123);

  auto arrow_array = ImportArray(&array, int32());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(int32(), "[123]")));
}

TEST(ArrayTest, ArrayTestAppendToInt16Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT16), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, std::numeric_limits<int64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 1);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const int16_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 123);

  auto arrow_array = ImportArray(&array, int16());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(int16(), "[123]")));
}

TEST(ArrayTest, ArrayTestAppendToInt8Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_INT8), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, std::numeric_limits<int64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 1);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const int8_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 1);

  auto arrow_array = ImportArray(&array, int8());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(int8(), "[1]")));
}

TEST(ArrayTest, ArrayTestAppendToStringArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, (5 + 1) * sizeof(int32_t));

  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("1234")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("56789")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto offset_buffer = reinterpret_cast<const int32_t*>(array.buffers[1]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[2]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  EXPECT_EQ(offset_buffer[0], 0);
  EXPECT_EQ(offset_buffer[1], 4);
  EXPECT_EQ(offset_buffer[2], 4);
  EXPECT_EQ(offset_buffer[3], 4);
  EXPECT_EQ(offset_buffer[4], 9);
  EXPECT_EQ(memcmp(data_buffer, "123456789", 9), 0);

  auto arrow_array = ImportArray(&array, utf8());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(utf8(), "[\"1234\", null, null, \"56789\"]")));
}

TEST(ArrayTest, ArrayTestAppendToUInt64Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_UINT64), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const uint64_t*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);

  auto arrow_array = ImportArray(&array, uint64());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(
      arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(uint64(), "[1, null, null, 3]")));
}

TEST(ArrayTest, ArrayTestAppendToUInt32Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_UINT32), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 3), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendInt(&array, -1), EINVAL);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 2);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const uint32_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 3);

  auto arrow_array = ImportArray(&array, uint32());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(uint32(), "[1, 3]")));
}

TEST(ArrayTest, ArrayTestAppendToUInt16Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_UINT16), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 3), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendInt(&array, -1), EINVAL);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 2);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const uint16_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 3);

  auto arrow_array = ImportArray(&array, uint16());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(uint16(), "[1, 3]")));
}

TEST(ArrayTest, ArrayTestAppendToUInt8Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_UINT8), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 3), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendInt(&array, -1), EINVAL);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 2);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const uint8_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 3);

  auto arrow_array = ImportArray(&array, uint8());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(uint8(), "[1, 3]")));
}

TEST(ArrayTest, ArrayTestAppendToDoubleArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_DOUBLE), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 3.14), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const double*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08 | 0x10);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);
  EXPECT_DOUBLE_EQ(data_buffer[4], 3.14);

  auto arrow_array = ImportArray(&array, float64());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(float64(), "[1.0, null, null, 3.0, 3.14]")));
}

TEST(ArrayTest, ArrayTestAppendToFloatArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_FLOAT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 3.14), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, std::numeric_limits<double>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const float*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08 | 0x10);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);
  EXPECT_FLOAT_EQ(data_buffer[4], 3.14);

  auto arrow_array = ImportArray(&array, float32());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(float32(), "[1.0, null, null, 3.0, 3.14]")));
}

TEST(ArrayTest, ArrayTestAppendToBoolArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_BOOL), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const uint8_t*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  EXPECT_EQ(ArrowBitGet(data_buffer, 0), 0x01);
  EXPECT_EQ(ArrowBitGet(data_buffer, 1), 0x00);
  EXPECT_EQ(ArrowBitGet(data_buffer, 2), 0x00);
  EXPECT_EQ(ArrowBitGet(data_buffer, 3), 0x00);

  auto arrow_array = ImportArray(&array, boolean());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(boolean(), "[true, null, null, false]")));
}

TEST(ArrayTest, ArrayTestAppendToLargeStringArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_LARGE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, (5 + 1) * sizeof(int64_t));

  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("1234")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("56789")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto offset_buffer = reinterpret_cast<const int64_t*>(array.buffers[1]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[2]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  EXPECT_EQ(offset_buffer[0], 0);
  EXPECT_EQ(offset_buffer[1], 4);
  EXPECT_EQ(offset_buffer[2], 4);
  EXPECT_EQ(offset_buffer[3], 4);
  EXPECT_EQ(offset_buffer[4], 9);
  EXPECT_EQ(memcmp(data_buffer, "123456789", 9), 0);

  auto arrow_array = ImportArray(&array, large_utf8());
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(large_utf8(), "[\"1234\", null, null, \"56789\"]")));
}

TEST(ArrayTest, ArrayTestAppendToFixedSizeBinaryArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ASSERT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 5),
            NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, 5 * 5);

  EXPECT_EQ(ArrowArrayAppendBytes(&array, {"12345", 5}), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendBytes(&array, {"67890", 5}), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 4);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0x01 | 0x08);
  char expected_data[] = {'1',  '2',  '3',  '4',  '5',  0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, '6',  '7',  '8',  '9',  '0'};
  EXPECT_EQ(memcmp(data_buffer, expected_data, 20), 0);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(fixed_size_binary(5), "[\"12345\", null, null, \"67890\"]")));
}

TEST(ArrayTest, ArrayTestAppendToListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve recursively without erroring
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(array.children[0], 1)->capacity_bytes, 0);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 456), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 789), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  // Make sure number of children is checked at finish
  array.n_children = 0;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected 1 child of list array but found 0 child arrays");
  array.n_children = 1;

  // Make sure final child size is checked at finish
  array.children[0]->length = array.children[0]->length - 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Expected child of list array with length >= 3 but found array with length 2");

  array.children[0]->length = array.children[0]->length + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(list(int64()), "[[123], null, [456, 789]]")));
}

TEST(ArrayTest, ArrayTestAppendToLargeListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_LARGE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve recursively without erroring
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(array.children[0], 1)->capacity_bytes, 0);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 456), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 789), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  // Make sure number of children is checked at finish
  array.n_children = 0;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected 1 child of large list array but found 0 child arrays");
  array.n_children = 1;

  // Make sure final child size is checked at finish
  array.children[0]->length = array.children[0]->length - 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected child of large list array with length >= 3 but found array with "
               "length 2");

  array.children[0]->length = array.children[0]->length + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(large_list(int64()), "[[123], null, [456, 789]]")));
}

TEST(ArrayTest, ArrayTestAppendToFixedSizeListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve recursively
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(array.children[0], 1)->capacity_bytes,
            2 * 5 * sizeof(int64_t));

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 456), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 789), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 12), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  // Make sure number of children is checked at finish
  array.n_children = 0;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected 1 child of fixed-size array but found 0 child arrays");
  array.n_children = 1;

  // Make sure final child size is checked at finish
  array.children[0]->length = array.children[0]->length - 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected child of fixed-size list array with length >= 6 but found array "
               "with length 5");

  array.children[0]->length = array.children[0]->length + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(
      ArrayFromJSON(fixed_size_list(int64(), 2), "[[123, 456], null, [789, 12]]")));
}

TEST(ArrayTest, ArrayTestAppendToStructArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "col1"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve recursively
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(array.children[0], 1)->capacity_bytes, 5 * sizeof(int64_t));

  // Wrong child length
  EXPECT_EQ(ArrowArrayFinishElement(&array), EINVAL);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 456), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(ArrayFromJSON(
      struct_({field("col1", int64())}), "[{\"col1\": 123}, null, {\"col1\": 456}]")));
}
