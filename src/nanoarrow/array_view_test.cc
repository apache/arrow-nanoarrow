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

#include "nanoarrow/nanoarrow.h"

TEST(ArrayTest, ArrayViewTestBasic) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_INT32);

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
  ArrowArrayInit(&array, NANOARROW_TYPE_INT32);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 11), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 12), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 13), NANOARROW_OK);
  array.length = 3;
  array.null_count = 0;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 3 * sizeof(int32_t));
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[0], 11);
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[1], 12);
  EXPECT_EQ(array_view.buffer_views[1].data.as_int32[2], 13);

  // Build with validity buffer
  ASSERT_EQ(ArrowBitmapAppend(ArrowArrayValidityBitmap(&array), 1, 3), NANOARROW_OK);
  array.null_count = -1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 3 * sizeof(int32_t));

  // Expect error for the wrong number of buffers
  ArrowArrayViewReset(&array_view);
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_STRING);
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), EINVAL);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestString) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_STRING);

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
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  array.null_count = 0;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  // Build non-zero length (the array ["abcd"])
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 0), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(&array, 1), 4), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 2), 4), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 2), "abcd", 4);
  array.length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (1 + 1) * sizeof(int32_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 4);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestLargeString) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_LARGE_STRING);

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
  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  array.null_count = 0;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 0);

  // Build non-zero length (the array ["abcd"])
  ASSERT_EQ(ArrowBufferAppendInt64(ArrowArrayBuffer(&array, 1), 0), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt64(ArrowArrayBuffer(&array, 1), 4), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferReserve(ArrowArrayBuffer(&array, 2), 4), NANOARROW_OK);
  ArrowBufferAppendUnsafe(ArrowArrayBuffer(&array, 2), "abcd", 4);
  array.length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 0);
  EXPECT_EQ(array_view.buffer_views[1].n_bytes, (1 + 1) * sizeof(int64_t));
  EXPECT_EQ(array_view.buffer_views[2].n_bytes, 4);

  array.release(&array);
  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestStruct) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_STRUCT);

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_STRUCT);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);

  // Exepct error for out-of-memory
  EXPECT_EQ(
      ArrowArrayViewAllocateChildren(&array_view, std::numeric_limits<int64_t>::max()),
      ENOMEM);

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 2), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 2);
  ArrowArrayViewInit(array_view.children[0], NANOARROW_TYPE_INT32);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);
  ArrowArrayViewInit(array_view.children[1], NANOARROW_TYPE_NA);
  EXPECT_EQ(array_view.children[1]->storage_type, NANOARROW_TYPE_NA);

  ArrowArrayViewSetLength(&array_view, 5);
  EXPECT_EQ(array_view.buffer_views[0].n_bytes, 1);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, 5 * sizeof(int32_t));

  // Exepct error for attempting to allocate a children array that already exists
  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), EINVAL);

  ArrowArrayViewReset(&array_view);
}

TEST(ArrayTest, ArrayViewTestFixedSizeList) {
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, NANOARROW_TYPE_FIXED_SIZE_LIST);
  array_view.layout.child_size_elements = 3;

  EXPECT_EQ(array_view.array, nullptr);
  EXPECT_EQ(array_view.storage_type, NANOARROW_TYPE_FIXED_SIZE_LIST);
  EXPECT_EQ(array_view.layout.buffer_type[0], NANOARROW_BUFFER_TYPE_VALIDITY);
  EXPECT_EQ(array_view.layout.element_size_bits[0], 1);

  EXPECT_EQ(ArrowArrayViewAllocateChildren(&array_view, 1), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  ArrowArrayViewInit(array_view.children[0], NANOARROW_TYPE_INT32);
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

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_STRUCT), NANOARROW_OK);

  // Expect error for the wrong number of children
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), EINVAL);

  ASSERT_EQ(ArrowArrayAllocateChildren(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInit(array.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  // Expect error for the wrong number of child elements
  array.length = 1;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), EINVAL);

  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 123),
            NANOARROW_OK);
  array.children[0]->length = 1;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
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

  ASSERT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 3),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewInitFromSchema(&array_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(array_view.n_children, 1);
  EXPECT_EQ(array_view.children[0]->storage_type, NANOARROW_TYPE_INT32);

  ASSERT_EQ(ArrowArrayInit(&array, NANOARROW_TYPE_FIXED_SIZE_LIST), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAllocateChildren(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInit(array.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  // Expect error for the wrong number of child elements
  array.length = 1;
  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), EINVAL);

  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 123),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 456),
            NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendInt32(ArrowArrayBuffer(array.children[0], 1), 789),
            NANOARROW_OK);
  array.children[0]->length = 3;
  ASSERT_EQ(ArrowArrayFinishBuilding(&array, false), NANOARROW_OK);

  EXPECT_EQ(ArrowArrayViewSetArray(&array_view, &array), NANOARROW_OK);
  EXPECT_EQ(array_view.children[0]->buffer_views[1].n_bytes, 3 * sizeof(int32_t));
  EXPECT_EQ(array_view.children[0]->buffer_views[1].data.as_int32[0], 123);

  ArrowArrayViewReset(&array_view);
  schema.release(&schema);
  array.release(&array);
}
