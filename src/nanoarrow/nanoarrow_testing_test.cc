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

ArrowErrorCode WriteSchemaJSON(std::ostream& out, const ArrowSchema* schema,
                               ArrowArrayView* array_view) {
  TestingJSONWriter writer;
  return writer.WriteSchema(out, schema);
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

void TestWriteJSON(std::function<ArrowErrorCode(ArrowSchema*)> type_expr,
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
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": "colname", "count": 0})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "VALIDITY": [], "DATA": []})");

  // Without a null value
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteColumnJSON,
      R"({"name": null, "count": 0, "VALIDITY": [], "children": []})");

  // Non-empty struct
  TestWriteJSON(
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
  TestWriteJSON(
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
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteBatchJSON,
      R"({"count": 0, "columns": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestSchema) {
  // Zero fields
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteSchemaJSON,
      R"({"fields": [], "metadata": null})");

  // More than zero fields
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 2));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_STRING));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteSchemaJSON,
      R"({"fields": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, "children": [], "metadata": null}], )"
      R"("metadata": null})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldBasic) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        schema->flags = 0;
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": false, "type": {"name": "null"}, "children": [], "metadata": null})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": "colname", "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldMetadata) {
  // Non-null but zero-size metadata
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetMetadata(schema, "\0\0\0\0"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": []})");

  // Non-zero size metadata
  TestWriteJSON(
      [](ArrowSchema* schema) {
        nanoarrow::UniqueBuffer buffer;
        NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderInit(buffer.get(), nullptr));
        NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderAppend(
            buffer.get(), ArrowCharView("k1"), ArrowCharView("v1")));
        NANOARROW_RETURN_NOT_OK(ArrowMetadataBuilderAppend(
            buffer.get(), ArrowCharView("k2"), ArrowCharView("v2")));

        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetMetadata(schema, reinterpret_cast<const char*>(buffer->data)));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], )"
      R"("metadata": [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldNested) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 2));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_STRING));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "struct"}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], "metadata": null}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, "children": [], "metadata": null}], )"
      R"("metadata": null})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypePrimitive) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "null"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BOOL);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "bool"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT8);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": true})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_UINT8);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": false})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_HALF_FLOAT);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "HALF"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "SINGLE"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_DOUBLE);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "DOUBLE"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "utf8"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largeutf8"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "binary"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_BINARY);
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largebinary"})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypeParameterized) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 123));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "fixedsizebinary", "byteWidth": 123})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeDecimal(schema, NANOARROW_TYPE_DECIMAL128, 10, 3));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "decimal", "bitWidth": 128, "precision": 10, "scale": 3})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "struct"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "list"})");

  TestWriteJSON(
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

  TestWriteJSON(
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

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "largelist"})");

  TestWriteJSON(
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
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypeUnion) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_SPARSE_UNION, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "SPARSE", "typeIds": []})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_SPARSE_UNION, 2));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_STRING));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "SPARSE", "typeIds": [0,1]})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_DENSE_UNION, 0));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "DENSE", "typeIds": []})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_DENSE_UNION, 2));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_STRING));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteTypeJSON,
      R"({"name": "union", "mode": "DENSE", "typeIds": [0,1]})");
}
