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
#include <math.h>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/builder_union.h>
#include <arrow/c/bridge.h>
#include <arrow/compare.h>

#include "nanoarrow/nanoarrow.h"

using namespace arrow;

// Lightweight versions of ArrowTesting's ARROW_EXPECT_OK. This
// version accomplishes the task of making sure the status message
// ends up in the ctests log.
void ARROW_EXPECT_OK(Status status) {
  if (!status.ok()) {
    throw std::runtime_error(status.message());
  }
}

void ARROW_EXPECT_OK(Result<std::shared_ptr<Array>> result) {
  if (!result.ok()) {
    throw std::runtime_error(result.status().message());
  }
}

TEST(ArrayTest, ArrayTestInit) {
  struct ArrowArray array;

  EXPECT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_UNINITIALIZED), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 0);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 1);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 2);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(array.n_buffers, 3);
  array.release(&array);

  EXPECT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_DATE64), EINVAL);
}

TEST(ArrayTest, ArrayTestAllocateChildren) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 0), NANOARROW_OK);
  EXPECT_EQ(array.n_children, 0);
  array.release(&array);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(
                &array, std::numeric_limits<int64_t>::max() / sizeof(void*)),
            ENOMEM);
  array.release(&array);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 2), NANOARROW_OK);
  EXPECT_EQ(array.n_children, 2);
  ASSERT_NE(array.children, nullptr);
  ASSERT_NE(array.children[0], nullptr);
  ASSERT_NE(array.children[1], nullptr);

  ASSERT_EQ(ArrowArrayInitFromType(array.children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(array.children[1], NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAllocateChildren(&array, 0), EINVAL);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestAllocateDictionary) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAllocateDictionary(&array), NANOARROW_OK);
  ASSERT_NE(array.dictionary, nullptr);

  ASSERT_EQ(ArrowArrayInitFromType(array.dictionary, NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  EXPECT_EQ(ArrowArrayAllocateDictionary(&array), EINVAL);

  array.release(&array);
}

TEST(ArrayTest, ArrayTestInitFromSchema) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 2), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema.children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema.children[1], NANOARROW_TYPE_STRING),
            NANOARROW_OK);

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
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
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
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
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
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);

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
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_NA), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 3);
  EXPECT_EQ(array.null_count, 3);

  auto arrow_array = ImportArray(&array, null());
  ARROW_EXPECT_OK(arrow_array);
  auto expected_array = MakeArrayOfNull(null(), 3);
  ARROW_EXPECT_OK(expected_array);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_NA), NANOARROW_OK);
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

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT64), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const int64_t*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0b00011001);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);
  EXPECT_EQ(data_buffer[4], 0);

  auto arrow_array = ImportArray(&array, int64());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = Int64Builder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(3));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToInt32Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
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

  auto builder = Int32Builder();
  ARROW_EXPECT_OK(builder.Append(123));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToInt16Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT16), NANOARROW_OK);
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

  auto builder = Int16Builder();
  ARROW_EXPECT_OK(builder.Append(123));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToInt8Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT8), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, std::numeric_limits<int64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, std::numeric_limits<uint64_t>::max()), EINVAL);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 1);
  EXPECT_EQ(array.null_count, 0);
  auto data_buffer = reinterpret_cast<const int8_t*>(array.buffers[1]);
  EXPECT_EQ(array.buffers[0], nullptr);
  EXPECT_EQ(data_buffer[0], 123);

  auto arrow_array = ImportArray(&array, int8());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = Int8Builder();
  ARROW_EXPECT_OK(builder.Append(123));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToStringArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, (5 + 1) * sizeof(int32_t));

  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("1234")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("56789")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto offset_buffer = reinterpret_cast<const int32_t*>(array.buffers[1]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[2]);
  EXPECT_EQ(validity_buffer[0], 0b00011001);
  EXPECT_EQ(offset_buffer[0], 0);
  EXPECT_EQ(offset_buffer[1], 4);
  EXPECT_EQ(offset_buffer[2], 4);
  EXPECT_EQ(offset_buffer[3], 4);
  EXPECT_EQ(offset_buffer[4], 9);
  EXPECT_EQ(memcmp(data_buffer, "123456789", 9), 0);

  auto arrow_array = ImportArray(&array, utf8());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = StringBuilder();
  ARROW_EXPECT_OK(builder.Append("1234"));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append("56789"));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendEmptyToString) {
  struct ArrowArray array;
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);
  EXPECT_NE(array.buffers[2], nullptr);
  array.release(&array);
}

TEST(ArrayTest, ArrayTestAppendToUInt64Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_UINT64), NANOARROW_OK);
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

  auto builder = UInt64Builder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(3));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToUInt32Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_UINT32), NANOARROW_OK);
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

  auto builder = UInt32Builder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.Append(3));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToUInt16Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_UINT16), NANOARROW_OK);
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

  auto builder = UInt16Builder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.Append(3));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToUInt8Array) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_UINT8), NANOARROW_OK);
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

  auto builder = UInt8Builder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.Append(3));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToDoubleArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_DOUBLE), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 3.14), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, std::numeric_limits<double>::max()),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, NAN), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, INFINITY), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, -INFINITY), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, -1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 11);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const double*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0b11111001);
  EXPECT_EQ(validity_buffer[1], 0b00000111);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);
  EXPECT_DOUBLE_EQ(data_buffer[4], 3.14);
  EXPECT_FLOAT_EQ(data_buffer[4], 3.14);
  EXPECT_FLOAT_EQ(data_buffer[5], std::numeric_limits<double>::max());
  EXPECT_TRUE(std::isnan(data_buffer[6])) << data_buffer[6];
  EXPECT_FLOAT_EQ(data_buffer[7], INFINITY);
  EXPECT_FLOAT_EQ(data_buffer[8], -INFINITY);
  EXPECT_FLOAT_EQ(data_buffer[9], -1);
  EXPECT_FLOAT_EQ(data_buffer[10], 0);

  auto arrow_array = ImportArray(&array, float64());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = DoubleBuilder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(3));
  ARROW_EXPECT_OK(builder.Append(3.14));
  ARROW_EXPECT_OK(builder.Append(std::numeric_limits<double>::max()));
  ARROW_EXPECT_OK(builder.Append(NAN));
  ARROW_EXPECT_OK(builder.Append(INFINITY));
  ARROW_EXPECT_OK(builder.Append(-INFINITY));
  ARROW_EXPECT_OK(builder.Append(-1));
  ARROW_EXPECT_OK(builder.Append(0));
  auto expected_array = builder.Finish();

  auto options = arrow::EqualOptions::Defaults().nans_equal(true);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe(), options));
}

TEST(ArrayTest, ArrayTestAppendToFloatArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_FLOAT), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendUInt(&array, 3), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 3.14), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, std::numeric_limits<double>::max()),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, NAN), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, INFINITY), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, -INFINITY), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, -1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendDouble(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 11);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const float*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0b11111001);
  EXPECT_EQ(validity_buffer[1], 0b00000111);
  EXPECT_EQ(data_buffer[0], 1);
  EXPECT_EQ(data_buffer[1], 0);
  EXPECT_EQ(data_buffer[2], 0);
  EXPECT_EQ(data_buffer[3], 3);
  EXPECT_FLOAT_EQ(data_buffer[4], 3.14);
  EXPECT_FLOAT_EQ(data_buffer[5], std::numeric_limits<float>::max());
  EXPECT_TRUE(std::isnan(data_buffer[6])) << data_buffer[6];
  EXPECT_FLOAT_EQ(data_buffer[7], INFINITY);
  EXPECT_FLOAT_EQ(data_buffer[8], -INFINITY);
  EXPECT_FLOAT_EQ(data_buffer[9], -1);
  EXPECT_FLOAT_EQ(data_buffer[10], 0);

  auto arrow_array = ImportArray(&array, float32());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = FloatBuilder();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(3));
  ARROW_EXPECT_OK(builder.Append(3.14));
  ARROW_EXPECT_OK(builder.Append(std::numeric_limits<double>::max()));
  ARROW_EXPECT_OK(builder.Append(NAN));
  ARROW_EXPECT_OK(builder.Append(INFINITY));
  ARROW_EXPECT_OK(builder.Append(-INFINITY));
  ARROW_EXPECT_OK(builder.Append(-1));
  ARROW_EXPECT_OK(builder.Append(0));
  auto expected_array = builder.Finish();

  auto options = arrow::EqualOptions::Defaults().nans_equal(true);
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe(), options));
}

TEST(ArrayTest, ArrayTestAppendToBoolArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_BOOL), NANOARROW_OK);
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

  auto builder = BooleanBuilder();
  ARROW_EXPECT_OK(builder.Append(true));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(false));
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToLargeStringArray) {
  struct ArrowArray array;

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_LARGE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, (5 + 1) * sizeof(int64_t));

  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("1234")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendString(&array, ArrowCharView("56789")), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto offset_buffer = reinterpret_cast<const int64_t*>(array.buffers[1]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[2]);
  EXPECT_EQ(validity_buffer[0], 0b00011001);
  EXPECT_EQ(offset_buffer[0], 0);
  EXPECT_EQ(offset_buffer[1], 4);
  EXPECT_EQ(offset_buffer[2], 4);
  EXPECT_EQ(offset_buffer[3], 4);
  EXPECT_EQ(offset_buffer[4], 9);
  EXPECT_EQ(memcmp(data_buffer, "123456789", 9), 0);

  auto arrow_array = ImportArray(&array, large_utf8());
  ARROW_EXPECT_OK(arrow_array);

  auto builder = LargeStringBuilder();
  ARROW_EXPECT_OK(builder.Append("1234"));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append("56789"));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToFixedSizeBinaryArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 5),
            NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(&array, 1)->capacity_bytes, 5 * 5);

  EXPECT_EQ(ArrowArrayAppendBytes(&array, {"12345", 5}), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendBytes(&array, {"67890", 5}), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(array.length, 5);
  EXPECT_EQ(array.null_count, 2);
  auto validity_buffer = reinterpret_cast<const uint8_t*>(array.buffers[0]);
  auto data_buffer = reinterpret_cast<const char*>(array.buffers[1]);
  EXPECT_EQ(validity_buffer[0], 0b00011001);
  char expected_data[] = {'1',  '2',  '3',  '4',  '5',  0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, '6',  '7',  '8',
                          '9',  '0',  0x00, 0x00, 0x00, 0x00, 0x00};
  EXPECT_EQ(memcmp(data_buffer, expected_data, 25), 0);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);

  auto builder = FixedSizeBinaryBuilder(fixed_size_binary(5));
  ARROW_EXPECT_OK(builder.Append("12345"));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append("67890"));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
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

  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

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

  auto child_builder = std::make_shared<Int64Builder>();
  auto builder = ListBuilder(default_memory_pool(), child_builder, list(int64()));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(123));
  ARROW_EXPECT_OK(builder.AppendNull());
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(456));
  ARROW_EXPECT_OK(child_builder->Append(789));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToLargeListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_LARGE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
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

  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

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

  auto child_builder = std::make_shared<Int64Builder>();
  auto builder =
      LargeListBuilder(default_memory_pool(), child_builder, large_list(int64()));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(123));
  ARROW_EXPECT_OK(builder.AppendNull());
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(456));
  ARROW_EXPECT_OK(child_builder->Append(789));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToMapArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_MAP), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0]->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0]->children[1], NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  // Check that we can reserve recursively without erroring
  ASSERT_EQ(ArrowArrayReserve(&array, 5), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayBuffer(array.children[0], 1)->capacity_bytes, 0);

  struct ArrowStringView string_value = ArrowCharView("foobar");
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0]->children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[0]->children[1], string_value),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(array.children[0]), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

  // Make sure number of children is checked at finish
  array.n_children = 0;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected 1 child of map array but found 0 child arrays");
  array.n_children = 1;

  // Make sure final child size is checked at finish
  array.children[0]->length = array.children[0]->length - 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Expected child of map array with length >= 1 but found array with length 0");

  array.children[0]->length = array.children[0]->length + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, &error), NANOARROW_OK);

  auto maybe_arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(maybe_arrow_array);
  auto arrow_array = std::move(maybe_arrow_array).MoveValueUnsafe();

  auto key_builder = std::make_shared<Int32Builder>();
  auto value_builder = std::make_shared<StringBuilder>();
  auto builder =
      MapBuilder(default_memory_pool(), key_builder, value_builder, map(int32(), utf8()));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(key_builder->Append(123));
  ARROW_EXPECT_OK(value_builder->Append("foobar"));
  ARROW_EXPECT_OK(builder.AppendNull());
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto maybe_expected_array = builder.Finish();
  ARROW_EXPECT_OK(maybe_expected_array);
  auto expected_array = std::move(maybe_expected_array).MoveValueUnsafe();

  EXPECT_TRUE(arrow_array->Equals(expected_array));
}

TEST(ArrayTest, ArrayTestAppendToFixedSizeListArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
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

  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

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
               "Expected child of fixed-size list array with length >= 8 but found array "
               "with length 7");

  array.children[0]->length = array.children[0]->length + 1;
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);

  auto child_builder = std::make_shared<Int64Builder>();
  auto builder = FixedSizeListBuilder(default_memory_pool(), child_builder,
                                      fixed_size_list(int64(), 2));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(123));
  ARROW_EXPECT_OK(child_builder->Append(456));
  ARROW_EXPECT_OK(builder.AppendNull());
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(child_builder->Append(789));
  ARROW_EXPECT_OK(child_builder->Append(12));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToStructArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeStruct(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
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

  ASSERT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);

  auto child_builder = std::make_shared<Int64Builder>();
  auto builder = StructBuilder(struct_({field("col1", int64())}), default_memory_pool(),
                               {child_builder});
  ARROW_EXPECT_OK(child_builder->Append(123));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(builder.AppendNull());
  ARROW_EXPECT_OK(child_builder->Append(456));
  ARROW_EXPECT_OK(builder.Append());
  ARROW_EXPECT_OK(builder.AppendEmptyValue());
  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestUnionUtils) {
  // Check length calculation with nullptr
  EXPECT_EQ(_ArrowParseUnionTypeIds("", nullptr), 0);
  EXPECT_EQ(_ArrowParseUnionTypeIds("0", nullptr), 1);
  EXPECT_EQ(_ArrowParseUnionTypeIds("0,1", nullptr), 2);
  // Invalid
  EXPECT_EQ(_ArrowParseUnionTypeIds("0,1,", nullptr), -1);
  EXPECT_EQ(_ArrowParseUnionTypeIds("0,1A", nullptr), -1);
  EXPECT_EQ(_ArrowParseUnionTypeIds("128", nullptr), -1);
  EXPECT_EQ(_ArrowParseUnionTypeIds("-1", nullptr), -1);

  // Check output
  int8_t type_ids[] = {-1, -1, -1, -1};
  EXPECT_EQ(_ArrowParseUnionTypeIds("4,5,6,7", type_ids), 4);
  EXPECT_EQ(type_ids[0], 4);
  EXPECT_EQ(type_ids[1], 5);
  EXPECT_EQ(type_ids[2], 6);
  EXPECT_EQ(type_ids[3], 7);

  // Check the "ids will equal child indices" checker
  EXPECT_TRUE(_ArrowUnionTypeIdsWillEqualChildIndices("", 0));
  EXPECT_TRUE(_ArrowUnionTypeIdsWillEqualChildIndices("0", 1));
  EXPECT_TRUE(_ArrowUnionTypeIdsWillEqualChildIndices("0,1", 2));

  EXPECT_FALSE(_ArrowUnionTypeIdsWillEqualChildIndices("0,1", 1));
  EXPECT_FALSE(_ArrowUnionTypeIdsWillEqualChildIndices(",", 0));
  EXPECT_FALSE(_ArrowUnionTypeIdsWillEqualChildIndices("1", 1));
  EXPECT_FALSE(_ArrowUnionTypeIdsWillEqualChildIndices("0,2", 2));
  EXPECT_FALSE(_ArrowUnionTypeIdsWillEqualChildIndices("0,2", 2));
}

TEST(ArrayTest, ArrayTestAppendToDenseUnionArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_DENSE_UNION, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "integers"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[1], "strings"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishUnionElement(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[1], ArrowCharView("one twenty four")),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishUnionElement(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);

  auto child_builder_int = std::make_shared<Int64Builder>();
  auto child_builder_string = std::make_shared<StringBuilder>();
  std::vector<std::shared_ptr<ArrayBuilder>> children = {child_builder_int,
                                                         child_builder_string};
  auto builder = DenseUnionBuilder(default_memory_pool(), children,
                                   arrow_array.ValueUnsafe()->type());
  ARROW_EXPECT_OK(builder.Append(0));
  ARROW_EXPECT_OK(child_builder_int->Append(123));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(child_builder_string->Append("one twenty four"));
  ARROW_EXPECT_OK(builder.AppendEmptyValue());

  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToSparseUnionArray) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_SPARSE_UNION, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT64), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "integers"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[1], "strings"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishUnionElement(&array, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendNull(&array, 2), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[1], ArrowCharView("one twenty four")),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishUnionElement(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendEmpty(&array, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  auto arrow_array = ImportArray(&array, &schema);
  ARROW_EXPECT_OK(arrow_array);

  auto child_builder_int = std::make_shared<Int64Builder>();
  auto child_builder_string = std::make_shared<StringBuilder>();
  std::vector<std::shared_ptr<ArrayBuilder>> children = {child_builder_int,
                                                         child_builder_string};
  auto builder = SparseUnionBuilder(default_memory_pool(), children,
                                    arrow_array.ValueUnsafe()->type());
  // Arrow's SparseUnionBuilder requires explicit empty value appends?
  ARROW_EXPECT_OK(builder.Append(0));
  ARROW_EXPECT_OK(child_builder_int->Append(123));
  ARROW_EXPECT_OK(child_builder_string->AppendEmptyValue());
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(child_builder_string->Append("one twenty four"));
  ARROW_EXPECT_OK(child_builder_int->AppendEmptyValue());
  ARROW_EXPECT_OK(builder.AppendEmptyValue());

  auto expected_array = builder.Finish();
  ARROW_EXPECT_OK(expected_array);

  EXPECT_EQ(arrow_array.ValueUnsafe()->ToString(),
            expected_array.ValueUnsafe()->ToString());
  EXPECT_TRUE(arrow_array.ValueUnsafe()->Equals(expected_array.ValueUnsafe()));
}

TEST(ArrayTest, ArrayTestAppendToUnionArrayErrors) {
  struct ArrowArray array;
  struct ArrowSchema schema;

  // Make an valid union that won't work during appending (because
  // the type_id != child_index)
  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_DENSE_UNION, 1),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+us:1"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(&array), EINVAL);
  schema.release(&schema);
  array.release(&array);
}

TEST(ArrayTest, ArrayViewTestBasic) {
  struct ArrowArrayView array_view;
  struct ArrowError error;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_INT32);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.buffer_type[1], NANOARROW_BUFFER_TYPE_DATA);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);
  EXPECT_EQ(array_view.layout.element_size_bits[1], 32);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 5 * sizeof(int32_t));

  struct ArrowArray array;

  // Build with no validity buffer
  ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 11), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 12), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 13), NANOARROW_OK);
  array.length = 3;
  array.null_count = 0;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 3 * sizeof(int32_t));
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[0], 11);
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[1], 12);
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[2], 13);

  // Build with validity buffer
  ASSERT_EQ(ArrowBitmapAppend(ArrowArrayValidityBitmap(&array), 1, 3), NANOARROW_OK);
  array.null_count = -1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 3 * sizeof(int32_t));

  // Expect error for the wrong number of buffers
  ArrowArrayViewReset(&array_view);
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_STRING);
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), EINVAL);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestMove) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_STRING);
  ASSERT_EQ(array_view.storage_type, NANOARROW_TYPE_STRING);

  struct ArrowArrayView array_view2;
  ArrowArrayViewInitFromType(&array_view2, NANOARROW_TYPE_UNINITIALIZED);
  ASSERT_EQ(array_view2.storage_type, NANOARROW_TYPE_UNINITIALIZED);

  ArrowArrayViewMove(&array_view, &array_view2);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_UNINITIALIZED);
  EXPECT_EQ(array_view2.storage_type, NANOARROW_TYPE_STRING);

  ArrowArrayViewReset(&array_view2);
}

TEST(ArrayTest, ArrayViewTestString) {
  struct ArrowArrayView array_view;
  struct ArrowError error;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_STRING);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_STRING);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.buffer_type[1], NANOARROW_BUFFER_TYPE_DATA_OFFSET);
  EXPECT_EQ(array_view.layout.buffer_type[2], NANOARROW_BUFFER_TYPE_DATA);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);
  EXPECT_EQ(array_view.layout.element_size_bits[1], 32);
  EXPECT_EQ(array_view.layout.element_size_bits[2], 0);

  // Can't assume offset buffer size > 0 if length == 0
  ArrowArrayViewSetLength(&array_view, 0);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (5 + 1) * sizeof(int32_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  struct ArrowArray array;

  // Build + check zero length
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  array.null_count = 0;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  // Build non-zero length (the array ["abcd"])
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 0), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 4), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 2), 4), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 2), "abcd", 4);
  array.length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (1 + 1) * sizeof(int32_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 4);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestLargeString) {
  struct ArrowArrayView array_view;
  struct ArrowError error;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_LARGE_STRING);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_LARGE_STRING);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.buffer_type[1], NANOARROW_BUFFER_TYPE_DATA_OFFSET);
  EXPECT_EQ(array_view.layout.buffer_type[2], NANOARROW_BUFFER_TYPE_DATA);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);
  EXPECT_EQ(array_view.layout.element_size_bits[1], 64);
  EXPECT_EQ(array_view.layout.element_size_bits[2], 0);

  // Can't assume offset buffer size > 0 if length == 0
  ArrowArrayViewSetLength(&array_view, 0);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (5 + 1) * sizeof(int64_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  struct ArrowArray array;

  // Build + check zero length
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  array.null_count = 0;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  // Build non-zero length (the array ["abcd"])
  ASSERT_EQ(ArrowBufferAppendInt64(ArrowArrayBuffer(&array, 1), 0), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt64(ArrowArrayBuffer(&array, 1), 4), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 2), 4), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 2), "abcd", 4);
  array.length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (1 + 1) * sizeof(int64_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 4);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestStruct) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_STRUCT);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_STRUCT);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);

  // Exepct error for out-of-memory
  EXPECT_EQ(ArrowArrayViewAllocateChildren(
                &array_view, std::numeric_limits<int64_t>::max() / sizeof(void*)),
            ENOMEM);

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 2), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 2);
  ArrowArrayViewInitFromType(array_view.children[0], NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);
  ArrowArrayViewInitFromType(array_view.children[1], NANOARROW_TYPE_NA);
  EXPECT_EQ(array_view.children[1]->storage_type, NANOARROW_TYPE_NA);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, 5 * sizeof(int32_t));

  // Exepct error for attempting to allocate a children array that already exists
  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), EINVAL);

  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestList) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_LIST);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_LIST);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);
  EXPECT_EQ(array_view.layout.buffer_type[1], NANOARROW_BUFFER_TYPE_DATA_OFFSET);
  EXPECT_EQ(array_view.layout.element_size_bits[1], 8 * sizeof(int32_t));

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  ArrowArrayViewInitFromType(array_view.children[0], NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (5 + 1) * sizeof(int32_t));

  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestLargeList) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_LARGE_LIST);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_LARGE_LIST);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);
  EXPECT_EQ(array_view.layout.buffer_type[1], NANOARROW_BUFFER_TYPE_DATA_OFFSET);
  EXPECT_EQ(array_view.layout.element_size_bits[1], 8 * sizeof(int64_t));

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  ArrowArrayViewInitFromType(array_view.children[0], NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (5 + 1) * sizeof(int64_t));

  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestFixedSizeList) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_FIXED_SIZE_LIST);
  array_view.layout.child_size_elements = 3;

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_FIXED_SIZE_LIST);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  ArrowArrayViewInitFromType(array_view.children[0], NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, 15 * sizeof(int32_t));

  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestStructArray) {
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema.children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);

  // Expect error for the wrong number of children
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), EINVAL);

  ASSERT_EQ(ArrowArrayAllocateChildren(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(array.children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);

  // Expect error for the wrong number of child elements
  array.length = 1;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), EINVAL);

  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 123),
            NANOARROW_OK);
  array.children[0]->length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, sizeof(int32_t));
  EXPECT_EQ(array_view.children[0]->buffer_views[1].data.as_int32[0], 123);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}

TEST(ArrayTest, ArrayViewTestFixedSizeListArray) {
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 3),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_FIXED_SIZE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAllocateChildren(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(array.children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);

  // Expect error for the wrong number of child elements
  array.length = 1;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), EINVAL);

  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 123),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 456),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 789),
            NANOARROW_OK);
  array.children[0]->length = 3;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, &error), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, 3 * sizeof(int32_t));
  EXPECT_EQ(array_view.children[0]->buffer_views[1].data.as_int32[0], 123);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}

TEST(ArrayTest, ArrayViewTestUnionChildIndices) {
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  struct ArrowSchema schema;

  // Build a simple union with one int and one string
  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_DENSE_UNION, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[1], ArrowCharView("one twenty four")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  // The ArrayView for a union could in theroy be created without a schema,
  // in which case the type_ids are assumed to equal child indices
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_DENSE_UNION);
  ASSERT_EQ(ArrowArrayViewAllocateChildren(&array_view, 2), NANOARROW_OK);
  ArrowArrayViewInitFromType(array_view.children[0], NANOARROW_TYPE_INT32);
  ArrowArrayViewInitFromType(array_view.children[1], NANOARROW_TYPE_STRING);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 1), 1);

  ArrowArrayViewReset(&array_view);

  // The test schema explicitly sets the type_ids 0,1 and this should work too
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 1), 1);

  ArrowArrayViewReset(&array_view);

  // Reversing the type ids should result in the same type ids but
  // reversed child indices
  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+ud:1,0"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, nullptr), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionTypeId(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 0), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 1), 0);

  ArrowArrayViewReset(&array_view);

  // Check the raw mapping in the array view for numbers that are easier to check
  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+ud:6,2"), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(array_view.union_type_id_map[6], 0);
  EXPECT_EQ(array_view.union_type_id_map[2], 1);
  EXPECT_EQ(array_view.union_type_id_map[128 + 0], 6);
  EXPECT_EQ(array_view.union_type_id_map[128 + 1], 2);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}

TEST(ArrayTest, ArrayViewTestDenseUnionGet) {
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  struct ArrowSchema schema;

  // Build a simple union with one int and one string and one null int
  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_DENSE_UNION, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[1], ArrowCharView("one twenty four")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  // Initialize the array view
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, nullptr), NANOARROW_OK);

  // Check the values that will be used to index into children
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 2), 0);

  // Check the values that will be used to index into the child arrays
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 1), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 2), 1);

  // Union elements are "never null" (even if the corresponding child element is)
  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 2), NANOARROW_OK);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}

TEST(ArrayTest, ArrayViewTestSparseUnionGet) {
  struct ArrowArrayView array_view;
  struct ArrowArray array;
  struct ArrowSchema schema;

  // Build a simple union with one int and one string and one null int
  ArrowSchemaInit(&schema);
  ASSERT_EQ(ArrowSchemaSetTypeUnion(&schema, NANOARROW_TYPE_SPARSE_UNION, 2),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[1], NANOARROW_TYPE_STRING), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(array.children[1], ArrowCharView("one twenty four")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishUnionElement(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, nullptr), NANOARROW_OK);

  // Initialize the array view
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, nullptr), NANOARROW_OK);

  // Check the values that will be used to index into children
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildIndex(&array_view, 2), 0);

  // Check the values that will be used to index into the child arrays
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewUnionChildOffset(&array_view, 2), 2);

  // Union elements are "never null" (even if the corresponding child element is)
  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 2), NANOARROW_OK);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}

template <typename TypeClass>
void TestGetFromNumericArrayView() {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowArrayView array_view;
  struct ArrowError error;

  auto data_type = TypeTraits<TypeClass>::type_singleton();

  // Array with nulls
  auto builder = NumericBuilder<TypeClass>();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append(4));
  auto maybe_arrow_array = builder.Finish();
  ARROW_EXPECT_OK(maybe_arrow_array);
  auto arrow_array = maybe_arrow_array.ValueUnsafe();

  ARROW_EXPECT_OK(ExportArray(*arrow_array, &array, &schema));
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 2), 1);
  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 3), 0);

  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(&array_view, 3), 4);
  EXPECT_EQ(ArrowArrayViewGetUIntUnsafe(&array_view, 3), 4);
  EXPECT_EQ(ArrowArrayViewGetDoubleUnsafe(&array_view, 3), 4.0);

  auto string_view = ArrowArrayViewGetStringUnsafe(&array_view, 0);
  EXPECT_EQ(string_view.data, nullptr);
  EXPECT_EQ(string_view.n_bytes, 0);
  auto buffer_view = ArrowArrayViewGetBytesUnsafe(&array_view, 0);
  EXPECT_EQ(buffer_view.data.data, nullptr);
  EXPECT_EQ(buffer_view.n_bytes, 0);

  ArrowArrayViewReset(&array_view);
  array.release(&array);
  schema.release(&schema);

  // Array without nulls (Arrow does not allocate the validity buffer)
  builder = NumericBuilder<TypeClass>();
  ARROW_EXPECT_OK(builder.Append(1));
  ARROW_EXPECT_OK(builder.Append(2));
  maybe_arrow_array = builder.Finish();
  ARROW_EXPECT_OK(maybe_arrow_array);
  arrow_array = maybe_arrow_array.ValueUnsafe();

  ARROW_EXPECT_OK(ExportArray(*arrow_array, &array, &schema));
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);

  // We're trying to test behavior with no validity buffer, so make sure that's true
  ASSERT_EQ(array_view.buffer_views[0].data.data, nullptr);

  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 1), 0);

  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(&array_view, 0), 1);
  EXPECT_EQ(ArrowArrayViewGetUIntUnsafe(&array_view, 1), 2);

  ArrowArrayViewReset(&array_view);
  array.release(&array);
  schema.release(&schema);
}

TEST(ArrayViewTest, ArrayViewTestGetNumeric) {
  TestGetFromNumericArrayView<Int64Type>();
  TestGetFromNumericArrayView<UInt64Type>();
  TestGetFromNumericArrayView<Int32Type>();
  TestGetFromNumericArrayView<UInt32Type>();
  TestGetFromNumericArrayView<Int16Type>();
  TestGetFromNumericArrayView<UInt32Type>();
  TestGetFromNumericArrayView<Int8Type>();
  TestGetFromNumericArrayView<UInt32Type>();
  TestGetFromNumericArrayView<DoubleType>();
  TestGetFromNumericArrayView<FloatType>();
}

template <typename BuilderClass>
void TestGetFromBinary(BuilderClass& builder) {
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowArrayView array_view;
  struct ArrowError error;

  auto data_type = builder.type();
  ARROW_EXPECT_OK(builder.Append("1234"));
  ARROW_EXPECT_OK(builder.AppendNulls(2));
  ARROW_EXPECT_OK(builder.Append("four"));
  auto maybe_arrow_array = builder.Finish();
  ARROW_EXPECT_OK(maybe_arrow_array);
  auto arrow_array = maybe_arrow_array.ValueUnsafe();

  ARROW_EXPECT_OK(ExportArray(*arrow_array, &array, &schema));
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(&array_view, &array, &error), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 2), 1);
  EXPECT_EQ(ArrowArrayViewIsNull(&array_view, 3), 0);

  auto string_view = ArrowArrayViewGetStringUnsafe(&array_view, 3);
  EXPECT_EQ(string_view.n_bytes, strlen("four"));
  EXPECT_EQ(memcmp(string_view.data, "four", string_view.n_bytes), 0);

  auto buffer_view = ArrowArrayViewGetBytesUnsafe(&array_view, 3);
  EXPECT_EQ(buffer_view.n_bytes, strlen("four"));
  EXPECT_EQ(memcmp(buffer_view.data.as_char, "four", buffer_view.n_bytes), 0);

  ArrowArrayViewReset(&array_view);
  array.release(&array);
  schema.release(&schema);
}

TEST(ArrayViewTest, ArrayViewTestGetString) {
  auto string_builder = StringBuilder();
  TestGetFromBinary<StringBuilder>(string_builder);

  auto binary_builder = BinaryBuilder();
  TestGetFromBinary<BinaryBuilder>(binary_builder);

  auto large_string_builder = LargeStringBuilder();
  TestGetFromBinary<LargeStringBuilder>(large_string_builder);

  auto large_binary_builder = LargeBinaryBuilder();
  TestGetFromBinary<LargeBinaryBuilder>(large_binary_builder);

  auto fixed_size_builder = FixedSizeBinaryBuilder(fixed_size_binary(4));
  TestGetFromBinary<FixedSizeBinaryBuilder>(fixed_size_builder);
}
