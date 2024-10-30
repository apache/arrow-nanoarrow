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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

using testing::ElementsAre;

TEST(HppArrayStream, EmptyArrayStream) {
  nanoarrow::UniqueSchema schema;
  struct ArrowArray array;

  nanoarrow::UniqueSchema schema_in;
  EXPECT_EQ(ArrowSchemaInitFromType(schema_in.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);

  nanoarrow::UniqueArrayStream array_stream;
  nanoarrow::EmptyArrayStream(schema_in.get()).ToArrayStream(array_stream.get());

  EXPECT_EQ(ArrowArrayStreamGetSchema(array_stream.get(), schema.get(), nullptr),
            NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  EXPECT_EQ(ArrowArrayStreamGetNext(array_stream.get(), &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.release, nullptr);
  EXPECT_STREQ(ArrowArrayStreamGetLastError(array_stream.get()), "");
}

TEST(HppArrayStream, VectorArrayStream) {
  nanoarrow::UniqueArray array_in;
  EXPECT_EQ(ArrowArrayInitFromType(array_in.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayStartAppending(array_in.get()), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayAppendInt(array_in.get(), 1234), NANOARROW_OK);
  EXPECT_EQ(ArrowArrayFinishBuildingDefault(array_in.get(), nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema schema_in;
  EXPECT_EQ(ArrowSchemaInitFromType(schema_in.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);

  nanoarrow::UniqueArrayStream array_stream;
  nanoarrow::VectorArrayStream(schema_in.get(), array_in.get())
      .ToArrayStream(array_stream.get());

  nanoarrow::ViewArrayStream array_stream_view(array_stream.get());
  for (ArrowArray& array : array_stream_view) {
    EXPECT_THAT(nanoarrow::ViewArrayAs<int32_t>(&array), ElementsAre(1234));
  }
  EXPECT_EQ(array_stream_view.count(), 1);
  EXPECT_EQ(array_stream_view.code(), NANOARROW_OK);
  EXPECT_STREQ(array_stream_view.error()->message, "");
}
