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

#include "nanoarrow/nanoarrow.hpp"

TEST(HppUnique, UniqueArray) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(array->release, nullptr);

  ASSERT_EQ(ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.get(), 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  EXPECT_NE(array->release, nullptr);
  EXPECT_EQ(array->length, 1);

  // move constructor
  nanoarrow::UniqueArray array2 = std::move(array);
  EXPECT_EQ(array->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_NE(array2->release, nullptr);
  EXPECT_EQ(array2->length, 1);

  // pointer constructor
  nanoarrow::UniqueArray array3(array2.get());
  EXPECT_EQ(array2->release, nullptr);
  EXPECT_NE(array3->release, nullptr);
  EXPECT_EQ(array3->length, 1);
}

TEST(HppUnique, UniqueSchema) {
  nanoarrow::UniqueSchema schema;
  EXPECT_EQ(schema->release, nullptr);

  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_NE(schema->release, nullptr);
  EXPECT_STREQ(schema->format, "i");

  // move constructor
  nanoarrow::UniqueSchema schema2 = std::move(schema);
  EXPECT_EQ(schema->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_NE(schema2->release, nullptr);
  EXPECT_STREQ(schema2->format, "i");

  // pointer constructor
  nanoarrow::UniqueSchema schema3(schema2.get());
  EXPECT_EQ(schema2->release, nullptr);
  EXPECT_NE(schema3->release, nullptr);
  EXPECT_STREQ(schema3->format, "i");
}

TEST(HppUnique, UniqueArrayStream) {
  nanoarrow::UniqueSchema schema;
  schema->format = NULL;

  nanoarrow::UniqueArrayStream array_stream_default;
  EXPECT_EQ(array_stream_default->release, nullptr);

  nanoarrow::UniqueSchema schema_in;
  EXPECT_EQ(ArrowSchemaInitFromType(schema_in.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  nanoarrow::UniqueArrayStream array_stream;
  nanoarrow::EmptyArrayStream(schema_in.get()).ToArrayStream(array_stream.get());
  EXPECT_NE(array_stream->release, nullptr);
  EXPECT_EQ(ArrowArrayStreamGetSchema(array_stream.get(), schema.get(), nullptr),
            NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  schema.reset();
  schema->format = NULL;

  // move constructor
  nanoarrow::UniqueArrayStream array_stream2 = std::move(array_stream);
  EXPECT_EQ(array_stream->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_NE(array_stream2->release, nullptr);
  EXPECT_EQ(ArrowArrayStreamGetSchema(array_stream2.get(), schema.get(), nullptr),
            NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  schema.reset();
  schema->format = NULL;

  // pointer constructor
  nanoarrow::UniqueArrayStream array_stream3(array_stream2.get());
  EXPECT_EQ(array_stream2->release, nullptr);
  EXPECT_NE(array_stream3->release, nullptr);
  EXPECT_EQ(ArrowArrayStreamGetSchema(array_stream2.get(), schema.get(), nullptr),
            NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");

  // releasing should clear the release callback
  EXPECT_EQ(ArrowSchemaInitFromType(schema_in.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  nanoarrow::UniqueArrayStream array_stream4;
  nanoarrow::EmptyArrayStream(schema_in.get()).ToArrayStream(array_stream4.get());
  EXPECT_NE(array_stream4->release, nullptr);
  array_stream4->release(array_stream4.get());
  EXPECT_EQ(array_stream4->private_data, nullptr);
  EXPECT_EQ(array_stream4->release, nullptr);
}

TEST(HppUnique, UniqueBuffer) {
  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(buffer->data, nullptr);
  EXPECT_EQ(buffer->size_bytes, 0);

  ASSERT_EQ(ArrowBufferAppendFill(buffer.get(), 0xff, 123), NANOARROW_OK);
  EXPECT_NE(buffer->data, nullptr);
  EXPECT_EQ(buffer->size_bytes, 123);

  // move constructor
  nanoarrow::UniqueBuffer buffer2 = std::move(buffer);
  EXPECT_EQ(buffer->data, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_EQ(buffer->size_bytes, 0);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_NE(buffer2->data, nullptr);
  EXPECT_EQ(buffer2->size_bytes, 123);

  // pointer constructor
  nanoarrow::UniqueBuffer buffer3(buffer2.get());
  EXPECT_EQ(buffer2->data, nullptr);
  EXPECT_EQ(buffer2->size_bytes, 0);
  EXPECT_NE(buffer3->data, nullptr);
  EXPECT_EQ(buffer3->size_bytes, 123);
}

TEST(HppUnique, UniqueBitmap) {
  nanoarrow::UniqueBitmap bitmap;
  EXPECT_EQ(bitmap->buffer.data, nullptr);
  EXPECT_EQ(bitmap->size_bits, 0);

  ASSERT_EQ(ArrowBitmapAppend(bitmap.get(), true, 123), NANOARROW_OK);
  EXPECT_NE(bitmap->buffer.data, nullptr);
  EXPECT_EQ(bitmap->size_bits, 123);

  // move constructor
  nanoarrow::UniqueBitmap bitmap2 = std::move(bitmap);
  EXPECT_EQ(bitmap->buffer.data, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bitmap->size_bits, 0);          // NOLINT(clang-analyzer-cplusplus.Move)
  EXPECT_NE(bitmap2->buffer.data, nullptr);
  EXPECT_EQ(bitmap2->size_bits, 123);

  // pointer constructor
  nanoarrow::UniqueBitmap bitmap3(bitmap2.get());
  EXPECT_EQ(bitmap2->buffer.data, nullptr);
  EXPECT_EQ(bitmap2->size_bits, 0);
  EXPECT_NE(bitmap3->buffer.data, nullptr);
  EXPECT_EQ(bitmap3->size_bits, 123);
}

TEST(HppUnique, UniqueArrayView) {
  nanoarrow::UniqueArrayView array_view;
  EXPECT_EQ(array_view->storage_type, NANOARROW_TYPE_UNINITIALIZED);

  // Use an ArrayView with children, since an ArrayView with no children
  // doesn't hold any resources
  ArrowArrayViewInitFromType(array_view.get(), NANOARROW_TYPE_STRUCT);
  ASSERT_EQ(ArrowArrayViewAllocateChildren(array_view.get(), 2), NANOARROW_OK);
  EXPECT_EQ(array_view->storage_type, NANOARROW_TYPE_STRUCT);

  // move constructor
  nanoarrow::UniqueArrayView array_view2 = std::move(array_view);
  EXPECT_EQ(array_view->storage_type,  // NOLINT(clang-analyzer-cplusplus.Move)
            NANOARROW_TYPE_UNINITIALIZED);
  EXPECT_EQ(array_view2->storage_type, NANOARROW_TYPE_STRUCT);

  // pointer constructor
  nanoarrow::UniqueArrayView array_view3(array_view2.get());
  EXPECT_EQ(array_view2->storage_type, NANOARROW_TYPE_UNINITIALIZED);
  EXPECT_EQ(array_view3->storage_type, NANOARROW_TYPE_STRUCT);
}
