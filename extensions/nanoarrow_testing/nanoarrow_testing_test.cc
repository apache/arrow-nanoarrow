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

#include <functional>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "nanoarrow_testing.hpp"

using nanoarrow::testing::TestingJSON;

void TestColumnPrimitive(ArrowType type, const char* field_name,
                         std::function<ArrowErrorCode(ArrowArray*)> append_expr,
                         const std::string& expected_json) {
  std::stringstream ss;

  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), type), NANOARROW_OK);
  if (field_name != nullptr) {
    ASSERT_EQ(ArrowSchemaSetName(schema.get(), field_name), NANOARROW_OK);
  }

  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(append_expr(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr), NANOARROW_OK);

  ASSERT_EQ(TestingJSON::WriteColumn(ss, schema.get(), array_view.get()), NANOARROW_OK);
  EXPECT_EQ(ss.str(), expected_json);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnNull) {
  TestColumnPrimitive(
      NANOARROW_TYPE_NA, nullptr, [](ArrowArray* array) { return NANOARROW_OK; },
      R"({"name": null, "count": 0})");

  TestColumnPrimitive(
      NANOARROW_TYPE_NA, "colname", [](ArrowArray* array) { return NANOARROW_OK; },
      R"({"name": "colname", "count": 0})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt) {
  TestColumnPrimitive(
      NANOARROW_TYPE_INT32, nullptr, [](ArrowArray* array) { return NANOARROW_OK; },
      R"({"name": null, "count": 0, "VALIDITY": [], "DATA": []})");

  // Without a null value
  TestColumnPrimitive(
      NANOARROW_TYPE_INT32, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": [0, 1, 0]})");

  // With two null values
  TestColumnPrimitive(
      NANOARROW_TYPE_INT32, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 2));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 3, "VALIDITY": [0, 0, 1], "DATA": [0, 0, 1]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt64) {
  TestColumnPrimitive(
      NANOARROW_TYPE_INT64, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnUInt64) {
  TestColumnPrimitive(
      NANOARROW_TYPE_UINT64, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnFloat) {
  TestColumnPrimitive(
      NANOARROW_TYPE_FLOAT, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 0.1234));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 1.2345));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], "DATA": [0.123, 1.235]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnString) {
  TestColumnPrimitive(
      NANOARROW_TYPE_STRING, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": [0, 3, 6], "DATA": ["abc", "def"]})");

  // Check a string that requires escaping of characters \ and "
  TestColumnPrimitive(
      NANOARROW_TYPE_STRING, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView(R"("\)")));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 1, "VALIDITY": [1], )"
      R"("OFFSET": [0, 2], "DATA": ["\"\\"]})");

  // Check a string that requires unicode escape
  TestColumnPrimitive(
      NANOARROW_TYPE_STRING, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("\u0001")));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 1, "VALIDITY": [1], )"
      R"("OFFSET": [0, 1], "DATA": ["\u0001"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnLargeString) {
  TestColumnPrimitive(
      NANOARROW_TYPE_LARGE_STRING, nullptr,
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": ["0", "3", "6"], "DATA": ["abc", "def"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnBinary) {
  TestColumnPrimitive(
      NANOARROW_TYPE_BINARY, nullptr,
      [](ArrowArray* array) {
        uint8_t value[] = {0x00, 0x01, 0xff};
        ArrowBufferView value_view;
        value_view.data.as_uint8 = value;
        value_view.size_bytes = sizeof(value);

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendBytes(array, value_view));
        return NANOARROW_OK;
      },
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": [0, 3, 6], "DATA": ["616263", "0001FF"]})");
}
