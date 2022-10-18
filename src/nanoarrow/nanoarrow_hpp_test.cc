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

TEST(NanoarrowHppTest, NanoarrowHppUniqueArrayTest) {
  nanoarrow::UniqueArray array;
  EXPECT_EQ(array->release, nullptr);

  ArrowArrayInit(array.get(), NANOARROW_TYPE_INT32);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.get(), 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuilding(array.get(), nullptr), NANOARROW_OK);

  EXPECT_NE(array->release, nullptr);
  EXPECT_EQ(array->length, 1);

  // move constructor
  nanoarrow::UniqueArray array2 = std::move(array);
  EXPECT_EQ(array->release, nullptr);
  EXPECT_NE(array2->release, nullptr);
  EXPECT_EQ(array2->length, 1);

  // pointer constructor
  nanoarrow::UniqueArray array3(array2.get());
  EXPECT_EQ(array2->release, nullptr);
  EXPECT_NE(array3->release, nullptr);
  EXPECT_EQ(array3->length, 1);
}

TEST(NanoarrowHppTest, NanoarrowHppUniqueSchemaTest) {
  nanoarrow::UniqueSchema schema;
  EXPECT_EQ(schema->release, nullptr);

  ArrowSchemaInit(schema.get(), NANOARROW_TYPE_INT32);
  EXPECT_NE(schema->release, nullptr);
  EXPECT_STREQ(schema->format, "i");

  // move constructor
  nanoarrow::UniqueSchema schema2 = std::move(schema);
  EXPECT_EQ(schema->release, nullptr);
  EXPECT_NE(schema2->release, nullptr);
  EXPECT_STREQ(schema2->format, "i");

  // pointer constructor
  nanoarrow::UniqueSchema schema3(schema2.get());
  EXPECT_EQ(schema2->release, nullptr);
  EXPECT_NE(schema3->release, nullptr);
  EXPECT_STREQ(schema3->format, "i");
}

TEST(NanoarrowHppTest, NanoarrowHppUniqueArrayStreamTest) {
  nanoarrow::UniqueSchema schema;
  schema->format = NULL;

  nanoarrow::UniqueArrayStream array_stream_default;
  EXPECT_EQ(array_stream_default->release, nullptr);

  auto array_stream = nanoarrow::EmptyArrayStream::MakeUnique(NANOARROW_TYPE_INT32);
  EXPECT_NE(array_stream->release, nullptr);
  EXPECT_EQ(array_stream.get_schema(schema.get()), NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  schema.release();
  schema->format = NULL;

  // move constructor
  nanoarrow::UniqueArrayStream array_stream2 = std::move(array_stream);
  EXPECT_EQ(array_stream->release, nullptr);
  EXPECT_NE(array_stream2->release, nullptr);
  EXPECT_EQ(array_stream2.get_schema(schema.get()), NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  schema.release();
  schema->format = NULL;

  // pointer constructor
  nanoarrow::UniqueArrayStream array_stream3(array_stream2.get());
  EXPECT_EQ(array_stream2->release, nullptr);
  EXPECT_NE(array_stream3->release, nullptr);
  EXPECT_EQ(array_stream3.get_schema(schema.get()), NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
}
