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

TEST(ArrayStreamTest, ArrayStreamTestBasic) {
  struct ArrowArrayStream array_stream;
  struct ArrowArray array;
  struct ArrowSchema schema;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(ArrowBasicArrayStreamInit(&array_stream, &schema, 1), NANOARROW_OK);
  EXPECT_EQ(schema.release, nullptr);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(&array, 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ArrowBasicArrayStreamSetArray(&array_stream, 0, &array);
  EXPECT_EQ(array.release, nullptr);

  EXPECT_EQ(ArrowBasicArrayStreamValidate(&array_stream, nullptr), NANOARROW_OK);

  struct ArrowSchema schema_copy;
  EXPECT_EQ(ArrowArrayStreamGetSchema(&array_stream, &schema_copy, nullptr),
            NANOARROW_OK);
  EXPECT_STREQ(schema_copy.format, "i");
  ArrowSchemaRelease(&schema_copy);

  nanoarrow::ViewArrayStream array_stream_view(&array_stream);
  for (ArrowArray& array : array_stream_view) {
    EXPECT_THAT(nanoarrow::ViewArrayAs<int32_t>(&array), ElementsAre(123));
    EXPECT_EQ(array.n_buffers, 2);
  }
  EXPECT_EQ(array_stream_view.count(), 1);
  EXPECT_EQ(array_stream_view.code(), NANOARROW_OK);
  EXPECT_STREQ(array_stream_view.error()->message, "");

  ArrowArrayStreamRelease(&array_stream);
  EXPECT_EQ(array_stream.release, nullptr);
}

TEST(ArrayStreamTest, ArrayStreamTestEmpty) {
  struct ArrowArrayStream array_stream;
  struct ArrowArray array;
  struct ArrowSchema schema;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  EXPECT_EQ(ArrowBasicArrayStreamInit(&array_stream, &schema, 0), NANOARROW_OK);
  EXPECT_EQ(ArrowBasicArrayStreamValidate(&array_stream, nullptr), NANOARROW_OK);

  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(ArrowArrayStreamGetNext(&array_stream, &array, nullptr), NANOARROW_OK);
    EXPECT_EQ(array.release, nullptr);
  }

  ArrowArrayStreamRelease(&array_stream);
}

TEST(ArrayStreamTest, ArrayStreamTestIncomplete) {
  struct ArrowArrayStream array_stream;
  struct ArrowArray array;
  struct ArrowSchema schema;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowBasicArrayStreamInit(&array_stream, &schema, 5), NANOARROW_OK);

  // Add five arrays with length == i
  for (int i = 0; i < 5; i++) {
    ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
    for (int j = 0; j < i; j++) {
      ASSERT_EQ(ArrowArrayAppendInt(&array, 123), NANOARROW_OK);
    }
    ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);
    ArrowBasicArrayStreamSetArray(&array_stream, i, &array);
  }

  // Pull only one of them
  EXPECT_EQ(ArrowArrayStreamGetNext(&array_stream, &array, nullptr), NANOARROW_OK);
  EXPECT_EQ(array.length, 0);
  ArrowArrayRelease(&array);

  // The remaining arrays, owned by the stream, should be released here
  ArrowArrayStreamRelease(&array_stream);
}

TEST(ArrayStreamTest, ArrayStreamTestInvalid) {
  struct ArrowArrayStream array_stream;
  struct ArrowArray array;
  struct ArrowSchema schema;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowBasicArrayStreamInit(&array_stream, &schema, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(&array_stream, 0, &array);

  EXPECT_EQ(ArrowBasicArrayStreamValidate(&array_stream, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected array with 2 buffer(s) but found 3 buffer(s)");

  ArrowArrayStreamRelease(&array_stream);
}
