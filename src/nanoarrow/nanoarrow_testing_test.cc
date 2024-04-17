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

#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow_testing.hpp"

using nanoarrow::testing::TestingJSONComparison;
using nanoarrow::testing::TestingJSONReader;
using nanoarrow::testing::TestingJSONWriter;

ArrowErrorCode WriteBatchJSON(std::ostream& out, TestingJSONWriter& writer,
                              const ArrowSchema* schema, ArrowArrayView* array_view) {
  return writer.WriteBatch(out, schema, array_view);
}

ArrowErrorCode WriteColumnJSON(std::ostream& out, TestingJSONWriter& writer,
                               const ArrowSchema* schema, ArrowArrayView* array_view) {
  return writer.WriteColumn(out, schema, array_view);
}

ArrowErrorCode WriteSchemaJSON(std::ostream& out, TestingJSONWriter& writer,
                               const ArrowSchema* schema, ArrowArrayView* array_view) {
  NANOARROW_UNUSED(array_view);
  return writer.WriteSchema(out, schema);
}

ArrowErrorCode WriteFieldJSON(std::ostream& out, TestingJSONWriter& writer,
                              const ArrowSchema* schema, ArrowArrayView* array_view) {
  NANOARROW_UNUSED(array_view);
  return writer.WriteField(out, schema);
}

ArrowErrorCode WriteTypeJSON(std::ostream& out, TestingJSONWriter& writer,
                             const ArrowSchema* schema, ArrowArrayView* array_view) {
  NANOARROW_UNUSED(array_view);
  return writer.WriteType(out, schema);
}

void TestWriteJSON(ArrowErrorCode (*type_expr)(ArrowSchema*),
                   ArrowErrorCode (*append_expr)(ArrowArray*),
                   ArrowErrorCode (*test_expr)(std::ostream&, TestingJSONWriter&,
                                               const ArrowSchema*, ArrowArrayView*),
                   const std::string& expected_json,
                   void (*setup_writer)(TestingJSONWriter& writer) = nullptr) {
  std::stringstream ss;

  nanoarrow::UniqueSchema schema;
  if (type_expr != nullptr) {
    ASSERT_EQ(type_expr(schema.get()), NANOARROW_OK);
  }

  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  if (append_expr != nullptr) {
    ASSERT_EQ(append_expr(array.get()), NANOARROW_OK);
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), nullptr), NANOARROW_OK);

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr), NANOARROW_OK);

  TestingJSONWriter writer;
  if (setup_writer != nullptr) {
    setup_writer(writer);
  }

  ASSERT_EQ(test_expr(ss, writer, schema.get(), array_view.get()), NANOARROW_OK);
  EXPECT_EQ(ss.str(), expected_json);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnNull) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      /*append_expr*/ nullptr, &WriteColumnJSON, R"({"name": null, "count": 0})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteColumnJSON, R"({"name": "colname", "count": 0})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnInt) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT32);
      },
      /*append_expr*/ nullptr, &WriteColumnJSON,
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
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnUInt64) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_UINT64);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array, 0));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": ["0", "1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnFloat) {
  // Test with constrained precision
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 0.1234));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 1.2345));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0.000, 0.123, 1.235]})",
      [](TestingJSONWriter& writer) { writer.set_float_precision(3); });

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 0.1234));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendDouble(array, 1.2345));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0.0, 0.1234000027179718, 1.2345000505447388]})",
      [](TestingJSONWriter& writer) { writer.set_float_precision(-1); });
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnString) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      [](ArrowArray* array) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], )"
      R"("OFFSET": [0, 0, 3, 6], "DATA": ["", "abc", "def"]})");

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
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("def")));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], )"
      R"("OFFSET": ["0", "0", "3", "6"], "DATA": ["", "abc", "def"]})");
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

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(array, ArrowCharView("abc")));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendBytes(array, value_view));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], )"
      R"("OFFSET": [0, 0, 3, 6], "DATA": ["", "616263", "0001FF"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnFixedSizeBinary) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        return ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 3);
      },
      [](ArrowArray* array) {
        uint8_t value[] = {0x00, 0x01, 0xff};
        ArrowBufferView value_view;
        value_view.data.as_uint8 = value;
        value_view.size_bytes = sizeof(value);

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array, 1));
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendBytes(array, value_view));
        return NANOARROW_OK;
      },
      &WriteColumnJSON,
      R"({"name": null, "count": 2, "VALIDITY": [0, 1], "DATA": ["000000", "0001FF"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestColumnStruct) {
  // Empty struct
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteColumnJSON,
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
      /*append_expr*/ nullptr, &WriteColumnJSON,
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
      /*append_expr*/ nullptr, &WriteColumnJSON,
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
      /*append_expr*/ nullptr, &WriteBatchJSON, R"({"count": 0, "columns": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestSchema) {
  // Zero fields
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteSchemaJSON, R"({"fields": []})");

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
      /*append_expr*/ nullptr, &WriteSchemaJSON,
      R"({"fields": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, "children": []}]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldBasic) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        schema->flags = 0;
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": false, "type": {"name": "null"}, "children": []})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetName(schema, "colname"));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": "colname", "nullable": true, "type": {"name": "null"}, "children": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldDict) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT16));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaAllocateDictionary(schema));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaInitFromType(schema->dictionary, NANOARROW_TYPE_STRING));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, )"
      R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 16, "isSigned": true}, )"
      R"("isOrdered": false}, "children": []})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldMetadata) {
  // Missing metadata
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})");

  // Non-null but zero-size metadata
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetMetadata(schema, "\0\0\0\0"));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteFieldJSON,
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
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], )"
      R"("metadata": [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}]})");

  // Ensure we can turn off metadata
  TestWriteJSON(
      [](ArrowSchema* schema) {
        NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA));
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetMetadata(schema, "\0\0\0\0"));
        return NANOARROW_OK;
      },
      [](ArrowArray* array) { return NANOARROW_OK; }, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})",
      [](TestingJSONWriter& writer) { writer.set_include_metadata(false); });
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
      /*append_expr*/ nullptr, &WriteFieldJSON,
      R"({"name": null, "nullable": true, "type": {"name": "struct"}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, "children": []}]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypePrimitive) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_NA);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "null"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BOOL);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "bool"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_INT8);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": true})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_UINT8);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "int", "bitWidth": 8, "isSigned": false})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_HALF_FLOAT);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "HALF"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_FLOAT);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "SINGLE"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_DOUBLE);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "floatingpoint", "precision": "DOUBLE"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRING);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "utf8"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_STRING);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "largeutf8"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_BINARY);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "binary"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        return ArrowSchemaInitFromType(schema, NANOARROW_TYPE_LARGE_BINARY);
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "largebinary"})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestTypeParameterized) {
  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 123));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "fixedsizebinary", "byteWidth": 123})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeDecimal(schema, NANOARROW_TYPE_DECIMAL128, 10, 3));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "decimal", "bitWidth": 128, "precision": 10, "scale": 3})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetTypeStruct(schema, 0));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "struct"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "list"})");

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
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "map", "keysSorted": false})");

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
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "map", "keysSorted": true})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_LIST));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON, R"({"name": "largelist"})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 12));
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
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
      /*append_expr*/ nullptr, &WriteTypeJSON,
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
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "union", "mode": "SPARSE", "typeIds": [0,1]})");

  TestWriteJSON(
      [](ArrowSchema* schema) {
        ArrowSchemaInit(schema);
        NANOARROW_RETURN_NOT_OK(
            ArrowSchemaSetTypeUnion(schema, NANOARROW_TYPE_DENSE_UNION, 0));
        return NANOARROW_OK;
      },
      /*append_expr*/ nullptr, &WriteTypeJSON,
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
      /*append_expr*/ nullptr, &WriteTypeJSON,
      R"({"name": "union", "mode": "DENSE", "typeIds": [0,1]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadSchema) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;

  ASSERT_EQ(
      reader.ReadSchema(
          R"({"fields": [)"
          R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})"
          R"(], "metadata": [{"key": "k1", "value": "v1"}]})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->format, "+s");
  ASSERT_EQ(schema->n_children, 1);
  EXPECT_STREQ(schema->children[0]->format, "n");

  ArrowMetadataReader metadata_reader;
  ASSERT_EQ(ArrowMetadataReaderInit(&metadata_reader, schema->metadata), NANOARROW_OK);
  ASSERT_EQ(metadata_reader.remaining_keys, 1);
  ArrowStringView key;
  ArrowStringView value;
  ASSERT_EQ(ArrowMetadataReaderRead(&metadata_reader, &key, &value), NANOARROW_OK);
  ASSERT_EQ(std::string(key.data, key.size_bytes), "k1");
  ASSERT_EQ(std::string(value.data, value.size_bytes), "v1");

  // Check invalid JSON
  EXPECT_EQ(reader.ReadSchema(R"({)", schema.get()), EINVAL);

  // Check at least one failed Check()
  EXPECT_EQ(reader.ReadSchema(R"("this is not a JSON object")", schema.get()), EINVAL);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadFieldBasic) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;

  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->format, "n");
  EXPECT_EQ(schema->name, nullptr);
  EXPECT_TRUE(schema->flags & ARROW_FLAG_NULLABLE);
  EXPECT_EQ(schema->n_children, 0);
  EXPECT_EQ(schema->metadata, nullptr);

  // Check non-nullable
  schema.reset();
  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": false, "type": {"name": "null"}, "children": []})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_FALSE(schema->flags & ARROW_FLAG_NULLABLE);

  // Check with name
  schema.reset();
  ASSERT_EQ(
      reader.ReadField(
          R"({"name": "colname", "nullable": true, "type": {"name": "null"}, "children": []})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->name, "colname");

  // Check invalid JSON
  EXPECT_EQ(reader.ReadField(R"({)", schema.get()), EINVAL);

  // Check at least one failed Check()
  EXPECT_EQ(reader.ReadField(R"("this is not a JSON object")", schema.get()), EINVAL);

  // Check that field is validated
  EXPECT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "fixedsizebinary", "byteWidth": -1}, "children": []})",
          schema.get()),
      EINVAL);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadFieldMetadata) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;

  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": [], )"
          R"("metadata": [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}]})",
          schema.get()),
      NANOARROW_OK);

  ArrowMetadataReader metadata;
  ArrowStringView key;
  ArrowStringView value;

  ASSERT_EQ(ArrowMetadataReaderInit(&metadata, schema->metadata), NANOARROW_OK);
  ASSERT_EQ(metadata.remaining_keys, 2);

  ASSERT_EQ(ArrowMetadataReaderRead(&metadata, &key, &value), NANOARROW_OK);
  ASSERT_EQ(std::string(key.data, key.size_bytes), "k1");
  ASSERT_EQ(std::string(value.data, value.size_bytes), "v1");

  ASSERT_EQ(ArrowMetadataReaderRead(&metadata, &key, &value), NANOARROW_OK);
  ASSERT_EQ(std::string(key.data, key.size_bytes), "k2");
  ASSERT_EQ(std::string(value.data, value.size_bytes), "v2");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadFieldNested) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;

  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "struct"}, "children": [)"
          R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []}], )"
          R"("metadata": null})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->format, "+s");
  ASSERT_EQ(schema->n_children, 1);
  EXPECT_STREQ(schema->children[0]->format, "n");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadFieldDictionary) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;

  // Unordered
  ASSERT_EQ(
      reader.ReadField(
          R"({"name": "col1", "nullable": true, "type": {"name": "utf8"}, "children": [], )"
          R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 32, "isSigned": true}, "isOrdered": false}})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->format, "i");
  EXPECT_STREQ(schema->name, "col1");
  EXPECT_TRUE(schema->flags & ARROW_FLAG_NULLABLE);
  EXPECT_FALSE(schema->flags & ARROW_FLAG_DICTIONARY_ORDERED);
  ASSERT_NE(schema->dictionary, nullptr);
  EXPECT_STREQ(schema->dictionary->format, "u");
  EXPECT_EQ(schema->dictionary->name, nullptr);
  EXPECT_EQ(schema->dictionary->dictionary, nullptr);

  // Ordered
  schema.reset();
  ASSERT_EQ(
      reader.ReadField(
          R"({"name": "col1", "nullable": true, "type": {"name": "utf8"}, "children": [], )"
          R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 32, "isSigned": true}, "isOrdered": true}})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_TRUE(schema->flags & ARROW_FLAG_DICTIONARY_ORDERED);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestRoundtripDataFile) {
  nanoarrow::UniqueArrayStream stream;
  ArrowError error;
  error.message[0] = '\0';

  std::string data_file_json =
      R"({"schema": {"fields": [)"
      R"({"name": "col1", "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": "col2", "nullable": true, "type": {"name": "utf8"}, "children": []}]})"
      R"(, "batches": [)"
      R"({"count": 1, "columns": [)"
      R"({"name": "col1", "count": 1}, )"
      R"({"name": "col2", "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["abc"]}]}, )"
      R"({"count": 2, "columns": [)"
      R"({"name": "col1", "count": 2}, )"
      R"({"name": "col2", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 3, 5], "DATA": ["abc", "de"]}]})"
      R"(]})";

  TestingJSONReader reader;
  ASSERT_EQ(reader.ReadDataFile(data_file_json, stream.get(),
                                TestingJSONReader::kNumBatchReadAll, &error),
            NANOARROW_OK)
      << error.message;

  TestingJSONWriter writer;
  std::stringstream data_file_json_roundtrip;
  ASSERT_EQ(writer.WriteDataFile(data_file_json_roundtrip, stream.get()), NANOARROW_OK);
  EXPECT_EQ(data_file_json_roundtrip.str(), data_file_json);

  stream.reset();
  data_file_json_roundtrip.str("");

  // Check with zero batches
  std::string data_file_json_empty = R"({"schema": {"fields": []}, "batches": []})";
  ASSERT_EQ(reader.ReadDataFile(data_file_json_empty, stream.get(),
                                TestingJSONReader::kNumBatchReadAll, &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(writer.WriteDataFile(data_file_json_roundtrip, stream.get()), NANOARROW_OK);
  EXPECT_EQ(data_file_json_roundtrip.str(), data_file_json_empty);

  // Also test error for invalid JSON
  ASSERT_EQ(reader.ReadDataFile("{", stream.get()), EINVAL);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestRoundtripDataFileDictionary) {
  nanoarrow::UniqueArrayStream stream;
  ArrowError error;
  error.message[0] = '\0';

  std::string data_file_json =
      R"({"schema": {"fields": [{"name": null, "nullable": true, "type": {"name": "binary"}, )"
      R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 32, "isSigned": true}, "isOrdered": false}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, )"
      R"("dictionary": {"id": 1, "indexType": {"name": "int", "bitWidth": 8, "isSigned": true}, "isOrdered": false}, "children": []}]}, )"
      R"("batches": [{"count": 1, "columns": [{"name": null, "count": 1, "VALIDITY": [1], "DATA": [0]}, )"
      R"({"name": null, "count": 1, "VALIDITY": [1], "DATA": [1]}]}], )"
      R"("dictionaries": [{"id": 0, "data": {"count": 1, "columns": [{"name": null, "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["616263"]}]}}, )"
      R"({"id": 1, "data": {"count": 2, "columns": [{"name": null, "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 3, 6], "DATA": ["abc", "def"]}]}}]})";

  TestingJSONReader reader;
  ASSERT_EQ(reader.ReadDataFile(data_file_json, stream.get(),
                                TestingJSONReader::kNumBatchReadAll, &error),
            NANOARROW_OK)
      << error.message;

  TestingJSONWriter writer;
  std::stringstream data_file_json_roundtrip;
  ASSERT_EQ(writer.WriteDataFile(data_file_json_roundtrip, stream.get()), NANOARROW_OK);
  EXPECT_EQ(data_file_json_roundtrip.str(), data_file_json);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadBatch) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  ArrowError error;
  error.message[0] = '\0';

  TestingJSONReader reader;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(reader.ReadBatch(R"({"count": 1, "columns": [{"name": null, "count": 1}]})",
                             schema.get(), array.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_NE(array->release, nullptr);
  EXPECT_EQ(array->length, 1);
  ASSERT_EQ(array->n_children, 1);
  EXPECT_EQ(array->children[0]->length, 1);

  // Check invalid JSON
  EXPECT_EQ(reader.ReadBatch(R"({)", schema.get(), array.get()), EINVAL);

  // Check that field is validated
  EXPECT_EQ(reader.ReadBatch(R"({"count": 1, "columns": [{"name": null, "count": -1}]})",
                             schema.get(), array.get()),
            EINVAL);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestReadColumnBasic) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  ArrowError error;
  error.message[0] = '\0';

  TestingJSONReader reader;

  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})",
          schema.get()),
      NANOARROW_OK);

  ASSERT_EQ(reader.ReadColumn(R"({"name": null, "count": 2})", schema.get(), array.get(),
                              &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(array->length, 2);

  // Check invalid JSON
  EXPECT_EQ(reader.ReadColumn(R"({)", schema.get(), array.get()), EINVAL);

  // Check at least one failed Check()
  EXPECT_EQ(
      reader.ReadColumn(R"("this is not a JSON object")", schema.get(), array.get()),
      EINVAL);

  // Check at least one failed PrefixError()
  EXPECT_EQ(reader.ReadColumn(R"({"name": "colname", "count": "not an integer"})",
                              schema.get(), array.get(), &error),
            EINVAL);
  EXPECT_STREQ(error.message, "-> Column 'colname' count must be integer");

  // Check that field is validated
  EXPECT_EQ(
      reader.ReadColumn(R"({"name": null, "count": -1})", schema.get(), array.get()),
      EINVAL);
}

void TestFieldRoundtrip(const std::string& field_json,
                        const std::string& column_json = "") {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;
  TestingJSONWriter writer;
  ArrowError error;
  error.message[0] = '\0';

  ASSERT_EQ(reader.ReadField(field_json, schema.get(), &error), NANOARROW_OK)
      << "Error: " << error.message;

  std::stringstream json_roundtrip;
  ASSERT_EQ(writer.WriteField(json_roundtrip, schema.get()), NANOARROW_OK);
  EXPECT_EQ(json_roundtrip.str(), field_json);

  if (column_json == "") {
    return;
  }

  nanoarrow::UniqueArray array;
  ASSERT_EQ(reader.ReadColumn(column_json, schema.get(), array.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr), NANOARROW_OK);

  json_roundtrip.str("");
  ASSERT_EQ(writer.WriteColumn(json_roundtrip, schema.get(), array_view.get()),
            NANOARROW_OK);
  EXPECT_EQ(json_roundtrip.str(), column_json);
}

void TestTypeRoundtrip(const std::string& type_json,
                       const std::string& column_json = "") {
  std::stringstream field_json_builder;
  field_json_builder << R"({"name": null, "nullable": true, "type": )" << type_json
                     << R"(, "children": []})";
  TestFieldRoundtrip(field_json_builder.str(), column_json);
}

void TestFieldError(const std::string& field_json, const std::string& msg,
                    int code = EINVAL) {
  nanoarrow::UniqueSchema schema;
  TestingJSONReader reader;
  ArrowError error;
  error.message[0] = '\0';

  EXPECT_EQ(reader.ReadField(field_json, schema.get(), &error), code);
  EXPECT_EQ(std::string(error.message), msg);
}

void TestTypeError(const std::string& type_json, const std::string& msg,
                   int code = EINVAL) {
  std::stringstream field_json_builder;
  field_json_builder << R"({"name": null, "nullable": true, "type": )" << type_json
                     << R"(, "children": []})";
  TestFieldError(field_json_builder.str(), msg, code);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldNull) {
  TestTypeRoundtrip(R"({"name": "null"})", R"({"name": null, "count": 2})");

  TestTypeError(R"({"name": "an unsupported type"})",
                "Unsupported Type name: 'an unsupported type'", ENOTSUP);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldBool) {
  TestTypeRoundtrip(
      R"({"name": "bool"})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0, 1, 0]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldString) {
  TestTypeRoundtrip(
      R"({"name": "utf8"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "OFFSET": [0, 3, 3], "DATA": ["abc", ""]})");
  TestTypeRoundtrip(
      R"({"name": "largeutf8"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "OFFSET": ["0", "3", "3"], "DATA": ["abc", ""]})");
  TestTypeRoundtrip(
      R"({"name": "binary"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "OFFSET": [0, 3, 3], "DATA": ["00FFA0", ""]})");
  TestTypeRoundtrip(
      R"({"name": "largebinary"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "OFFSET": ["0", "3", "3"], "DATA": ["00FFA0", ""]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldInt) {
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 8, "isSigned": true})",
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": [-128, 0, 127]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 16, "isSigned": true})",
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": [-129, 0, 127]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 32, "isSigned": true})",
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": [-130, 0, 127]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 64, "isSigned": true})",
      R"({"name": null, "count": 3, "VALIDITY": [1, 1, 1], "DATA": ["-131", "0", "127"]})");

  TestTypeError(R"({"name": "int", "bitWidth": 1, "isSigned": true})",
                "Type[name=='int'] bitWidth must be 8, 16, 32, or 64");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldUInt) {
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 8, "isSigned": false})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0, 0, 255]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 16, "isSigned": false})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0, 0, 256]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 32, "isSigned": false})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0, 0, 257]})");
  TestTypeRoundtrip(
      R"({"name": "int", "bitWidth": 64, "isSigned": false})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": ["0", "0", "258"]})");

  TestTypeError(R"({"name": "int", "bitWidth": 1, "isSigned": false})",
                "Type[name=='int'] bitWidth must be 8, 16, 32, or 64");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldFloatingPoint) {
  TestTypeRoundtrip(R"({"name": "floatingpoint", "precision": "HALF"})");
  TestTypeRoundtrip(
      R"({"name": "floatingpoint", "precision": "SINGLE"})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0.0, 1.0, 2.0]})");
  TestTypeRoundtrip(
      R"({"name": "floatingpoint", "precision": "DOUBLE"})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": [0.0, 4.0, 5.0]})");

  TestTypeError(
      R"({"name": "floatingpoint", "precision": "NOT_A_PRECISION"})",
      "Type[name=='floatingpoint'] precision must be 'HALF', 'SINGLE', or 'DOUBLE'");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldFixedSizeBinary) {
  TestTypeRoundtrip(
      R"({"name": "fixedsizebinary", "byteWidth": 3})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["00FFA0", "000000"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldDecimal) {
  TestTypeRoundtrip(
      R"({"name": "decimal", "bitWidth": 128, "precision": 10, "scale": 3})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": ["0", "0", "258"]})");
  TestTypeRoundtrip(
      R"({"name": "decimal", "bitWidth": 256, "precision": 10, "scale": 3})",
      R"({"name": null, "count": 3, "VALIDITY": [0, 1, 1], "DATA": ["0", "0", "258"]})");

  TestTypeError(R"({"name": "decimal", "bitWidth": 123, "precision": 10, "scale": 3})",
                "Type[name=='decimal'] bitWidth must be 128 or 256");

  // Ensure that omitted bitWidth maps to decimal128
  TestingJSONReader reader;
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(
      reader.ReadField(
          R"({"name": null, "nullable": true, "type": {"name": "decimal", "precision": 10, "scale": 3}, "children": []})",
          schema.get()),
      NANOARROW_OK);
  EXPECT_STREQ(schema->format, "d:10,3");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldDate) {
  TestTypeRoundtrip(R"({"name": "date", "unit": "DAY"})",
                    R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [1, 0]})");

  TestTypeRoundtrip(
      R"({"name": "date", "unit": "MILLISECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["86400000", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldTime) {
  TestTypeRoundtrip(R"({"name": "time", "unit": "SECOND", "bitWidth": 32})",
                    R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [1, 0]})");
  TestTypeRoundtrip(R"({"name": "time", "unit": "MILLISECOND", "bitWidth": 32})",
                    R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [1, 0]})");
  TestTypeRoundtrip(
      R"({"name": "time", "unit": "MICROSECOND", "bitWidth": 64})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "time", "unit": "NANOSECOND", "bitWidth": 64})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldTimestamp) {
  TestTypeRoundtrip(
      R"({"name": "timestamp", "unit": "SECOND", "timezone": "America/Halifax"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");

  TestTypeRoundtrip(
      R"({"name": "timestamp", "unit": "SECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "timestamp", "unit": "MILLISECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "timestamp", "unit": "MICROSECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "timestamp", "unit": "NANOSECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldDuration) {
  TestTypeRoundtrip(
      R"({"name": "duration", "unit": "SECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "duration", "unit": "MILLISECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "duration", "unit": "MICROSECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
  TestTypeRoundtrip(
      R"({"name": "duration", "unit": "NANOSECOND"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": ["1", "0"]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldInterval) {
  TestTypeRoundtrip(R"({"name": "interval", "unit": "YEAR_MONTH"})",
                    R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [1, 0]})");

  TestTypeRoundtrip(
      R"({"name": "interval", "unit": "DAY_TIME"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [{"days": 1, "milliseconds": 2}, {"days": 0, "milliseconds": 0}]})");

  TestTypeRoundtrip(
      R"({"name": "interval", "unit": "MONTH_DAY_NANO"})",
      R"({"name": null, "count": 2, "VALIDITY": [1, 0], "DATA": [{"months": 1, "days": 2, "nanoseconds": "3"}, {"months": 0, "days": 0, "nanoseconds": "0"}]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldMap) {
  // Sorted keys
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "map", "keysSorted": true}, "children": [)"
      R"({"name": "entries", "nullable": false, "type": {"name": "struct"}, "children": [)"
      R"({"name": null, "nullable": false, "type": {"name": "utf8"}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "bool"}, "children": []})"
      R"(]})"
      R"(]})");

  // Unsorted keys
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "map", "keysSorted": false}, "children": [)"
      R"({"name": "entries", "nullable": false, "type": {"name": "struct"}, "children": [)"
      R"({"name": null, "nullable": false, "type": {"name": "utf8"}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "bool"}, "children": []})"
      R"(]})"
      R"(]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldStruct) {
  // Empty
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "struct"}, "children": [)"
      R"(]})",
      R"({"name": null, "count": 0, "VALIDITY": [], "children": []})");

  // Non-empty
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "struct"}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})"
      R"(]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldList) {
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "list"}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})"
      R"(]})");

  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "largelist"}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})"
      R"(]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldFixedSizeList) {
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "fixedsizelist", "listSize": 12}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []})"
      R"(]})");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldUnion) {
  // Empty unions
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "union", "mode": "DENSE", "typeIds": []}, "children": []})",
      R"({"name": null, "count": 0, "TYPE_ID": [], "OFFSET": [], "children": []})");
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "union", "mode": "SPARSE", "typeIds": []}, "children": []})",
      R"({"name": null, "count": 0, "TYPE_ID": [], "children": []})");

  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "union", "mode": "DENSE", "typeIds": [10,20]}, "children": [)"
      R"({"name": null, "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, "children": []})"
      R"(]})");

  // Non-empty unions (null, "abc")
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "union", "mode": "SPARSE", "typeIds": [10,20]}, "children": [)"
      R"({"name": "nulls", "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": "strings", "nullable": true, "type": {"name": "utf8"}, "children": []})"
      R"(]})",
      R"({"name": null, "count": 2, "TYPE_ID": [20, 10], "children": [)"
      R"({"name": "nulls", "count": 2}, )"
      R"({"name": "strings", "count": 2, "VALIDITY": [1, 1], "OFFSET": [0, 3, 3], "DATA": ["abc", ""]})"
      R"(]})");
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "union", "mode": "DENSE", "typeIds": [10,20]}, "children": [)"
      R"({"name": "nulls", "nullable": true, "type": {"name": "null"}, "children": []}, )"
      R"({"name": "strings", "nullable": true, "type": {"name": "utf8"}, "children": []})"
      R"(]})",
      R"({"name": null, "count": 2, "TYPE_ID": [20, 10], "OFFSET": [0, 0], "children": [)"
      R"({"name": "nulls", "count": 1}, )"
      R"({"name": "strings", "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["abc"]})"
      R"(]})");

  TestTypeError(R"({"name": "union", "mode": "NOT_A_MODE", "typeIds": []})",
                "Type[name=='union'] mode must be 'DENSE' or 'SPARSE'");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFieldDictionaryRoundtrip) {
  // Unordered
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, )"
      R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 16, "isSigned": true}, )"
      R"("isOrdered": false}, "children": []})");

  // Ordered
  TestFieldRoundtrip(
      R"({"name": null, "nullable": true, "type": {"name": "utf8"}, )"
      R"("dictionary": {"id": 0, "indexType": {"name": "int", "bitWidth": 16, "isSigned": true}, )"
      R"("isOrdered": true}, "children": []})");
}

void AssertSchemasCompareEqual(
    ArrowSchema* actual, ArrowSchema* expected,
    void (*setup_comparison)(TestingJSONComparison&) = nullptr) {
  TestingJSONComparison comparison;
  std::stringstream msg;

  if (setup_comparison != nullptr) {
    setup_comparison(comparison);
  }

  ASSERT_EQ(comparison.CompareSchema(actual, expected), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "");
}

void AssertSchemasCompareUnequal(
    ArrowSchema* actual, ArrowSchema* expected, int num_differences,
    const std::string& differences,
    void (*setup_comparison)(TestingJSONComparison&) = nullptr) {
  TestingJSONComparison comparison;
  std::stringstream msg;

  if (setup_comparison != nullptr) {
    setup_comparison(comparison);
  }

  ASSERT_EQ(comparison.CompareSchema(actual, expected), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), num_differences);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), differences);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestSchemaComparison) {
  nanoarrow::UniqueSchema actual;
  nanoarrow::UniqueSchema expected;

  // Start with two identical schemas and ensure there are no differences
  ArrowSchemaInit(actual.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(actual.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(actual->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaDeepCopy(actual.get(), expected.get()), NANOARROW_OK);

  AssertSchemasCompareEqual(actual.get(), expected.get());

  // With different top-level flags
  actual->flags = 0;
  AssertSchemasCompareUnequal(actual.get(), expected.get(), /*num_differences*/ 1,
                              "Path: \n- .flags: 0\n+ .flags: 2\n\n");
  // With different top-level flags but turning off that comparison
  AssertSchemasCompareEqual(actual.get(), expected.get(),
                            [](TestingJSONComparison& comparison) {
                              comparison.set_compare_batch_flags((false));
                            });
  actual->flags = expected->flags;

  // With different top-level metadata
  nanoarrow::UniqueBuffer buf;
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key1"),
                                       ArrowCharView("value1")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  AssertSchemasCompareUnequal(actual.get(), expected.get(), /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key1", "value": "value1"}, {"key": "key2", "value": "value2"}]
+ null

)");

  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), nullptr), NANOARROW_OK);

  // With different children
  actual->children[0]->flags = 0;
  AssertSchemasCompareUnequal(actual.get(), expected.get(), /*num_differences*/ 1,
                              /*differences*/ R"(Path: .children[0]
- {"name": null, "nullable": false, "type": {"name": "null"}, "children": []}
+ {"name": null, "nullable": true, "type": {"name": "null"}, "children": []}

)");
  actual->children[0]->flags = expected->children[0]->flags;

  // With different numbers of children
  actual.reset();
  ArrowSchemaInit(actual.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(actual.get(), 0), NANOARROW_OK);
  AssertSchemasCompareUnequal(
      actual.get(), expected.get(), /*num_differences*/ 1,
      /*differences*/ "Path: \n- .n_children: 0\n+ .n_children: 1\n\n");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestSchemaComparisonMap) {
  nanoarrow::UniqueSchema actual;
  nanoarrow::UniqueSchema expected;

  // Start with two identical schemas with maps and ensure there are no differences
  ArrowSchemaInit(actual.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(actual.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(actual->children[0], NANOARROW_TYPE_MAP), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(actual->children[0]->children[0]->children[0],
                               NANOARROW_TYPE_STRING),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(actual->children[0]->children[0]->children[1],
                               NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaDeepCopy(actual.get(), expected.get()), NANOARROW_OK);

  AssertSchemasCompareEqual(actual.get(), expected.get());

  // Even when one of the maps has different namees, there should be no differences
  ASSERT_EQ(
      ArrowSchemaSetName(actual->children[0]->children[0], "this name is not 'entries'"),
      NANOARROW_OK);
  AssertSchemasCompareEqual(actual.get(), expected.get());

  // This should also be true if the map is nested below the top-level of the schema
  nanoarrow::UniqueSchema actual2;
  ASSERT_EQ(ArrowSchemaInitFromType(actual2.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(actual2.get(), 1), NANOARROW_OK);
  ArrowSchemaMove(actual.get(), actual2->children[0]);
  expected.reset();
  ASSERT_EQ(ArrowSchemaDeepCopy(actual2.get(), expected.get()), NANOARROW_OK);
  ASSERT_EQ(
      ArrowSchemaSetName(expected->children[0]->children[0]->children[0], "entries"),
      NANOARROW_OK);

  AssertSchemasCompareEqual(actual2.get(), expected.get());
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestMetadataComparison) {
  nanoarrow::UniqueSchema actual;
  nanoarrow::UniqueSchema expected;
  nanoarrow::UniqueBuffer buf;

  // Start with two identical schemas and ensure there are no differences
  ArrowSchemaInit(actual.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(actual.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(actual->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaDeepCopy(actual.get(), expected.get()), NANOARROW_OK);
  AssertSchemasCompareEqual(actual.get(), expected.get());

  // With different top-level metadata that are not equivalent because of order
  buf.reset();
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key1"),
                                       ArrowCharView("value1")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  buf.reset();
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key1"),
                                       ArrowCharView("value1")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(expected.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  // ...using the comparison that considers ordering
  AssertSchemasCompareUnequal(actual.get(), expected.get(), /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key1", "value": "value1"}, {"key": "key2", "value": "value2"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)");

  // ...using the comparison that does *not* consider ordering
  AssertSchemasCompareEqual(actual.get(), expected.get(),
                            [](TestingJSONComparison& comparison) {
                              comparison.set_compare_metadata_order(false);
                            });

  // With different top-level metadata that are not equivalent because of number of items
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), nullptr), NANOARROW_OK);

  // ...using the comparison that considers ordering
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- null
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)",
                              [](TestingJSONComparison& comparison) {
                                comparison.set_compare_metadata_order(false);
                              });

  // ...using the comparison that does *not* consider ordering
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- null
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)",
                              [](TestingJSONComparison& comparison) {
                                comparison.set_compare_metadata_order(false);
                              });

  // With different top-level metadata that are not equivalent because of item content
  buf.reset();
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key1"),
                                       ArrowCharView("gazornenplat")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  // ...using the schema comparison that considers order
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "gazornenplat"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)");

  // ...and using the schema comparison that does *not* consider order
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "gazornenplat"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)",
                              [](TestingJSONComparison& comparison) {
                                comparison.set_compare_metadata_order(false);
                              });

  // With different top-level metadata that are not equivalent because of item keys
  buf.reset();
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key3"),
                                       ArrowCharView("value1")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  // ...using the schema comparison that considers order
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key2", "value": "value2"}, {"key": "key3", "value": "value1"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)");

  // ...and using the schema comparison that does *not* consider order
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key2", "value": "value2"}, {"key": "key3", "value": "value1"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)",
                              [](TestingJSONComparison& comparison) {
                                comparison.set_compare_metadata_order(false);
                              });

  // Metadata that are not equal and contain duplicate keys
  buf.reset();
  ASSERT_EQ(ArrowMetadataBuilderInit(buf.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowMetadataBuilderAppend(buf.get(), ArrowCharView("key2"),
                                       ArrowCharView("value2 again")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetMetadata(actual.get(), reinterpret_cast<char*>(buf->data)),
            NANOARROW_OK);

  // ...using the schema comparison that considers order
  AssertSchemasCompareUnequal(actual.get(), expected.get(),
                              /*num_differences*/ 1,
                              /*differences*/
                              "Path: .metadata"
                              R"(
- [{"key": "key2", "value": "value2"}, {"key": "key2", "value": "value2 again"}]
+ [{"key": "key2", "value": "value2"}, {"key": "key1", "value": "value1"}]

)");

  // Comparison is not implemented for the comparison that does not consider order
  TestingJSONComparison comparison;
  comparison.set_compare_metadata_order(false);
  ASSERT_EQ(comparison.CompareSchema(actual.get(), expected.get()), ENOTSUP);
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestArrayComparison) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray actual;
  nanoarrow::UniqueArray expected;
  TestingJSONComparison comparison;
  std::stringstream msg;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);
  ASSERT_EQ(comparison.SetSchema(schema.get()), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(actual.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(actual->children[0], 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(actual.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(expected.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(expected->children[0], 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(expected.get(), nullptr), NANOARROW_OK);

  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);
  comparison.ClearDifferences();

  actual->length = 1;
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "Path: \n- .length: 1\n+ .length: 0\n\n");
  msg.str("");
  comparison.ClearDifferences();
  actual->length = 0;

  actual->offset = 1;
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "Path: \n- .offset: 1\n+ .offset: 0\n\n");
  msg.str("");
  comparison.ClearDifferences();
  actual->offset = 0;

  actual->children[0]->length = 2;
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), R"(Path: .children[0]
- {"name": null, "count": 2}
+ {"name": null, "count": 1}

)");
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestFloatingPointArrayComparison) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray actual;
  nanoarrow::UniqueArray expected;
  TestingJSONComparison comparison;
  std::stringstream msg;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_DOUBLE), NANOARROW_OK);
  ASSERT_EQ(comparison.SetSchema(schema.get()), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(actual.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendDouble(actual->children[0], 1.23456789), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(actual.get(), nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(expected.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendDouble(expected->children[0], 1.23456), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(expected.get(), nullptr), NANOARROW_OK);

  // Default precision: all decimal places
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.ClearDifferences();

  // With just enough decimal places to trigger a difference
  comparison.set_compare_float_precision(5);
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.ClearDifferences();

  // With just few enough decimal places to be considered equivalent
  comparison.set_compare_float_precision(4);
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);
  comparison.ClearDifferences();
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestArrayWithDictionaryComparison) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray actual;
  nanoarrow::UniqueArray expected;

  TestingJSONComparison comparison;
  std::stringstream msg;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(schema->children[0]), NANOARROW_OK);
  ASSERT_EQ(
      ArrowSchemaInitFromType(schema->children[0]->dictionary, NANOARROW_TYPE_STRING),
      NANOARROW_OK);
  ASSERT_EQ(comparison.SetSchema(schema.get()), NANOARROW_OK);

  // Dictionary-encoded with one element
  ASSERT_EQ(ArrowArrayInitFromSchema(expected.get(), schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(expected.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(expected->children[0], 0), NANOARROW_OK);
  ASSERT_EQ(
      ArrowArrayAppendString(expected->children[0]->dictionary, ArrowCharView("abc")),
      NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(expected.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(expected.get(), nullptr), NANOARROW_OK);

  // Dictionary-encoded with one element with the only difference in the dictionary
  ASSERT_EQ(ArrowArrayInitFromSchema(actual.get(), schema.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(actual.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(actual->children[0], 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(actual->children[0]->dictionary, ArrowCharView("def")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(actual.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(actual.get(), nullptr), NANOARROW_OK);

  // Compare array with dictionary that has no differences
  ASSERT_EQ(comparison.CompareBatch(actual.get(), actual.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);
  comparison.ClearDifferences();

  // Compare arrays with nested difference in the dictionary
  ArrowError error;
  ASSERT_EQ(comparison.CompareBatch(actual.get(), expected.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), R"(Path: .children[0].dictionary
- {"name": null, "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["def"]}
+ {"name": null, "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["abc"]}

)");
}

ArrowErrorCode MakeArrayStream(const ArrowSchema* schema,
                               std::vector<std::string> batches_json,
                               ArrowArrayStream* out) {
  TestingJSONReader reader;
  nanoarrow::UniqueSchema schema_copy;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaDeepCopy(schema, schema_copy.get()));
  NANOARROW_RETURN_NOT_OK(
      ArrowBasicArrayStreamInit(out, schema_copy.get(), batches_json.size()));

  nanoarrow::UniqueArray array;
  for (size_t i = 0; i < batches_json.size(); i++) {
    NANOARROW_RETURN_NOT_OK(reader.ReadBatch(batches_json[i], schema, array.get()));
    ArrowBasicArrayStreamSetArray(out, i, array.get());
  }

  return NANOARROW_OK;
}

TEST(NanoarrowTestingTest, NanoarrowTestingTestArrayStreamComparison) {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArrayStream actual;
  nanoarrow::UniqueArrayStream expected;

  std::string null1_batch_json =
      R"({"count": 1, "columns": [{"name": null, "count": 1}]})";

  TestingJSONComparison comparison;
  std::stringstream msg;

  ArrowSchemaInit(schema.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);

  // Identical streams with 0 batches
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema.get(), {}, actual.get()), NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {}, expected.get()), NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);
  comparison.WriteDifferences(msg);

  // Identical streams with >0 batches
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema.get(), {null1_batch_json}, actual.get()),
            NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {null1_batch_json}, expected.get()),
            NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 0);

  // Stream where actual has more batches
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema.get(), {null1_batch_json}, actual.get()),
            NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {}, expected.get()), NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "Path: Batch 0\n- unfinished stream\n+ finished stream\n\n");
  msg.str("");
  comparison.ClearDifferences();

  // Stream where expected has more batches
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema.get(), {}, actual.get()), NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {null1_batch_json}, expected.get()),
            NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "Path: Batch 0\n- finished stream\n+ unfinished stream\n\n");
  msg.str("");
  comparison.ClearDifferences();

  // Stream where schemas differ
  nanoarrow::UniqueSchema schema2;
  ArrowSchemaInit(schema2.get());
  ASSERT_EQ(ArrowSchemaSetTypeStruct(schema2.get(), 2), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema2->children[0], NANOARROW_TYPE_NA), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema2->children[1], NANOARROW_TYPE_NA), NANOARROW_OK);
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema2.get(), {}, actual.get()), NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {}, expected.get()), NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), "Path: Schema\n- .n_children: 1\n+ .n_children: 2\n\n");
  msg.str("");
  comparison.ClearDifferences();

  // Stream where batches differ
  actual.reset();
  expected.reset();
  ASSERT_EQ(MakeArrayStream(schema.get(),
                            {R"({"count": 1, "columns": [{"name": null, "count": 2}]})"},
                            actual.get()),
            NANOARROW_OK);
  ASSERT_EQ(MakeArrayStream(schema.get(), {null1_batch_json}, expected.get()),
            NANOARROW_OK);
  ASSERT_EQ(comparison.CompareArrayStream(actual.get(), expected.get()), NANOARROW_OK);
  EXPECT_EQ(comparison.num_differences(), 1);
  comparison.WriteDifferences(msg);
  EXPECT_EQ(msg.str(), R"(Path: Batch 0.children[0]
- {"name": null, "count": 2}
+ {"name": null, "count": 1}

)");
}
