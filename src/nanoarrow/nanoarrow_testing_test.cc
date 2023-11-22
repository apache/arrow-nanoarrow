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

#include "nanoarrow/nanoarrow_testing.hpp"

using nanoarrow::testing::TestingJSONWriter;

ArrowErrorCode WriteBatchJSON(std::ostream& out, const ArrowSchema* schema,
                              ArrowArrayView* array_view) {
  TestingJSONWriter writer;
  return writer.WriteBatch(out, schema, array_view);
}

ArrowErrorCode WriteColumnJSON(std::ostream& out, const ArrowSchema* schema,
                               ArrowArrayView* array_view) {
  TestingJSONWriter writer;
  return writer.WriteColumn(out, schema, array_view);
}

ArrowErrorCode WriteFieldJSON(std::ostream& out, const ArrowSchema* schema,
                              ArrowArrayView* array_view) {
  TestingJSONWriter writer;
  return writer.WriteField(out, schema);
}

ArrowErrorCode WriteTypeJSON(std::ostream& out, const ArrowSchema* schema,
                             ArrowArrayView* array_view) {
  TestingJSONWriter writer;
  return writer.WriteType(out, schema);
}

void TestColumn(std::function<ArrowErrorCode(ArrowSchema*)> type_expr,
                std::function<ArrowErrorCode(ArrowArray*)> append_expr,
                ArrowErrorCode (*test_expr)(std::ostream&, const ArrowSchema*,
                                            ArrowArrayView*),
                const std::string& expected_json) {
  std::stringstream ss;

  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(type_expr(schema.get()), NANOARROW_OK);
  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(append_expr(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr), NANOARROW_OK);

  ASSERT_EQ(test_expr(ss, schema.get(), array_view.get()), NANOARROW_OK);
  EXPECT_EQ(ss.str(), expected_json);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnNull) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0})");

  TestColumn(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": "colname", "count": 0})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "VALIDITY": [], "DATA": []})");

  // Without a null value
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": [0, 1, 0]})");

  // With two null values
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 2));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 0, 1], "DATA": [0, 0, 1]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt64) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT64);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnUInt64) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_UINT64);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnFloat) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 0.1234));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 1.2345));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], "DATA": [0.123, 1.235]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnString) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": [0, 3, 6], "DATA": ["abc", "def"]})");

  // Check a string that requires escaping of characters \ and "
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView(R"("\)")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 1, "VALIDITY": [1], )"
      R"("OFFSET": [0, 2], "DATA": ["\"\\"]})");

  // Check a string that requires unicode escape
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("\u0001")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 1, "VALIDITY": [1], )"
      R"("OFFSET": [0, 1], "DATA": ["\u0001"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnLargeString) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": ["0", "3", "6"], "DATA": ["abc", "def"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnBinary) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
      },
      [](ArrowArray* array) {
        uint8_t value[] = {0x00, 0x01, 0xff};
        ArrowBufferView value_view;
        value_view.data.as_uint8 = value;
        value_view.size_bytes = sizeof(value);

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendBytes(array, value_view));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 2, "VALIDITY": [1, 1], )"
      R"("OFFSET": [0, 3, 6], "DATA": ["616263", "0001FF"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnStruct) {
  // Empty struct
  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "VALIDITY": [], "children": []})");

  // Non-empty struct
  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 2));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[0], "col1"));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema->children[1], "col2"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "VALIDITY": [], "children": [)"
      R"({"name": "col1", "count": 0}, {"name": "col2", "count": 0}]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnDenseUnion) {
  // Empty union
  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_DENSE_UNION, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "TYPE_ID": [], "OFFSET": [], "children": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestBatch) {
  // Empty batch
  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteBatchJSON,
      R"({"count": 0, "columns": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldBasic) {
  TestColumn(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null})");

  TestColumn(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        schema->flags = 0;
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": false, "type": {"name": "null"}, "children": [], "metadata": null})");

  TestColumn(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": "colname", "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldMetadata) {
  TestColumn(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypePrimitive) {
  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "null"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BOOL);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "bool"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT8);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": true})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_UINT8);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": false})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_HALF_FLOAT);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "HALF"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "SINGLE"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_DOUBLE);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "DOUBLE"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "utf8"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largeutf8"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "binary"})");

  TestColumn(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_BINARY);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largebinary"})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypeParameterized) {
  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 123));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "fixedsizebinary", "byteWidth": 123})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeDecimal(schema, NANOARROW_TYPE_DECIMAL128, 10, 3));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "decimal", "bitWidth": 128, "precision": 10, "scale": 3})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "struct"})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "list"})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_MAP));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0]->children[0], NANOARROW_TYPE_STRING));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0]->children[1], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "map", "keysSorted": false})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_MAP));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0]->children[0], NANOARROW_TYPE_STRING));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0]->children[1], NANOARROW_TYPE_INT32));
        schema->flags = ARROW_FLAG_MAP_KEYS_SORTED;
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "map", "keysSorted": true})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largelist"})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 12));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "fixedsizelist", "listSize": 12})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_SPARSE_UNION, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "SPARSE", "typeIds": []})");

  TestColumn(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_DENSE_UNION, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "DENSE", "typeIds": []})");
}
