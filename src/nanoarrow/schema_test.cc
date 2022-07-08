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

#include <arrow/c/bridge.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/key_value_metadata.h>

#include "nanoarrow/nanoarrow.h"

using namespace arrow;

TEST(SchemaTest, SchemaInit) {
  struct ArrowSchema schema;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 2), NANOARROW_OK);

  ASSERT_NE(schema.release, nullptr);
  EXPECT_EQ(schema.format, nullptr);
  EXPECT_EQ(schema.name, nullptr);
  EXPECT_EQ(schema.metadata, nullptr);
  EXPECT_EQ(schema.n_children, 2);
  EXPECT_EQ(schema.children[0]->release, nullptr);
  EXPECT_EQ(schema.children[1]->release, nullptr);

  schema.release(&schema);
  EXPECT_EQ(schema.release, nullptr);
}

static void ExpectSchemaInitOk(enum ArrowType data_type,
                               std::shared_ptr<DataType> expected_arrow_type) {
  struct ArrowSchema schema;
  EXPECT_EQ(ArrowSchemaInit(&schema, data_type), NANOARROW_OK);
  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(expected_arrow_type));
}

TEST(SchemaTest, SchemaInitSimple) {
  ExpectSchemaInitOk(NANOARROW_TYPE_NA, null());
  ExpectSchemaInitOk(NANOARROW_TYPE_BOOL, boolean());
  ExpectSchemaInitOk(NANOARROW_TYPE_UINT8, uint8());
  ExpectSchemaInitOk(NANOARROW_TYPE_INT8, int8());
  ExpectSchemaInitOk(NANOARROW_TYPE_UINT16, uint16());
  ExpectSchemaInitOk(NANOARROW_TYPE_INT16, int16());
  ExpectSchemaInitOk(NANOARROW_TYPE_UINT32, uint32());
  ExpectSchemaInitOk(NANOARROW_TYPE_INT32, int32());
  ExpectSchemaInitOk(NANOARROW_TYPE_UINT64, uint64());
  ExpectSchemaInitOk(NANOARROW_TYPE_INT64, int64());
  ExpectSchemaInitOk(NANOARROW_TYPE_HALF_FLOAT, float16());
  ExpectSchemaInitOk(NANOARROW_TYPE_FLOAT, float32());
  ExpectSchemaInitOk(NANOARROW_TYPE_DOUBLE, float64());
  ExpectSchemaInitOk(NANOARROW_TYPE_STRING, utf8());
  ExpectSchemaInitOk(NANOARROW_TYPE_LARGE_STRING, large_utf8());
  ExpectSchemaInitOk(NANOARROW_TYPE_BINARY, binary());
  ExpectSchemaInitOk(NANOARROW_TYPE_LARGE_BINARY, large_binary());
  ExpectSchemaInitOk(NANOARROW_TYPE_DATE32, date32());
  ExpectSchemaInitOk(NANOARROW_TYPE_DATE64, date64());
  ExpectSchemaInitOk(NANOARROW_TYPE_INTERVAL_MONTHS, month_interval());
  ExpectSchemaInitOk(NANOARROW_TYPE_INTERVAL_DAY_TIME, day_time_interval());
  ExpectSchemaInitOk(NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO, month_day_nano_interval());
}

TEST(SchemaTest, SchemaInitSimpleError) {
  struct ArrowSchema schema;
  EXPECT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_DECIMAL128), EINVAL);
  EXPECT_EQ(schema.release, nullptr);
}

TEST(SchemaTest, SchemaTestInitNestedList) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_LIST), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+l");
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(list(int32())));

  EXPECT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_LARGE_LIST), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+L");
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(large_list(int32())));
}

TEST(SchemaTest, SchemaTestInitNestedStruct) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+s");
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(struct_({field("item", int32())})));
}

TEST(SchemaTest, SchemaTestInitNestedMap) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_MAP), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+m");
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "entries"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema.children[0], 2), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0]->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0]->children[0], "key"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0]->children[1], NANOARROW_TYPE_STRING),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0]->children[1], "value"), NANOARROW_OK);

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(map(int32(), utf8())));
}

TEST(SchemaTest, SchemaInitFixedSize) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_DOUBLE, 1), EINVAL);
  EXPECT_EQ(schema.release, nullptr);
  EXPECT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 0),
            EINVAL);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, 45),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "w:45");

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(fixed_size_binary(45)));

  EXPECT_EQ(ArrowSchemaInitFixedSize(&schema, NANOARROW_TYPE_FIXED_SIZE_LIST, 12),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "+w:12");
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema.children[0], "item"), NANOARROW_OK);

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(fixed_size_list(int32(), 12)));
}

TEST(SchemaTest, SchemaInitDecimal) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInitDecimal(&schema, NANOARROW_TYPE_DECIMAL128, -1, 1), EINVAL);
  EXPECT_EQ(schema.release, nullptr);
  EXPECT_EQ(ArrowSchemaInitDecimal(&schema, NANOARROW_TYPE_DOUBLE, 1, 2), EINVAL);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitDecimal(&schema, NANOARROW_TYPE_DECIMAL128, 1, 2),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "d:1,2");

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(decimal128(1, 2)));

  EXPECT_EQ(ArrowSchemaInitDecimal(&schema, NANOARROW_TYPE_DECIMAL256, 3, 4),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "d:3,4,256");
  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(decimal256(3, 4)));
}

TEST(SchemaTest, SchemaInitDateTime) {
  struct ArrowSchema schema;

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_DOUBLE,
                                    NANOARROW_TIME_UNIT_SECOND, nullptr),
            EINVAL);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIME32,
                                    NANOARROW_TIME_UNIT_SECOND, "non-null timezone"),
            EINVAL);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_DURATION,
                                    NANOARROW_TIME_UNIT_SECOND, "non-null timezone"),
            EINVAL);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitDateTime(
                &schema, NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_SECOND,
                "a really really really really really really really really really really "
                "long timezone that causes a buffer overflow on snprintf"),
            ERANGE);
  EXPECT_EQ(schema.release, nullptr);

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIME32,
                                    NANOARROW_TIME_UNIT_SECOND, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tts");

  auto arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(time32(TimeUnit::SECOND)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIME64,
                                    NANOARROW_TIME_UNIT_NANO, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "ttn");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(time64(TimeUnit::NANO)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_DURATION,
                                    NANOARROW_TIME_UNIT_SECOND, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tDs");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(duration(TimeUnit::SECOND)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIMESTAMP,
                                    NANOARROW_TIME_UNIT_SECOND, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tss:");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(timestamp(TimeUnit::SECOND)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIMESTAMP,
                                    NANOARROW_TIME_UNIT_MILLI, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tsm:");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(timestamp(TimeUnit::MILLI)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIMESTAMP,
                                    NANOARROW_TIME_UNIT_MICRO, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tsu:");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(timestamp(TimeUnit::MICRO)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIMESTAMP,
                                    NANOARROW_TIME_UNIT_NANO, NULL),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tsn:");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(arrow_type.ValueUnsafe()->Equals(timestamp(TimeUnit::NANO)));

  EXPECT_EQ(ArrowSchemaInitDateTime(&schema, NANOARROW_TYPE_TIMESTAMP,
                                    NANOARROW_TIME_UNIT_SECOND, "America/Halifax"),
            NANOARROW_OK);
  EXPECT_STREQ(schema.format, "tss:America/Halifax");

  arrow_type = ImportType(&schema);
  ARROW_EXPECT_OK(arrow_type);
  EXPECT_TRUE(
      arrow_type.ValueUnsafe()->Equals(timestamp(TimeUnit::SECOND, "America/Halifax")));
}

TEST(SchemaTest, SchemaSetFormat) {
  struct ArrowSchema schema;
  ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED);

  EXPECT_EQ(ArrowSchemaSetFormat(&schema, "i"), NANOARROW_OK);
  EXPECT_STREQ(schema.format, "i");

  EXPECT_EQ(ArrowSchemaSetFormat(&schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(schema.format, nullptr);

  schema.release(&schema);
}

TEST(SchemaTest, SchemaSetName) {
  struct ArrowSchema schema;
  ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED);

  EXPECT_EQ(ArrowSchemaSetName(&schema, "a_name"), NANOARROW_OK);
  EXPECT_STREQ(schema.name, "a_name");

  EXPECT_EQ(ArrowSchemaSetName(&schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(schema.name, nullptr);

  schema.release(&schema);
}

TEST(SchemaTest, SchemaSetMetadata) {
  struct ArrowSchema schema;
  ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED);

  // (test will only work on little endian)
  char simple_metadata[] = {'\1', '\0', '\0', '\0', '\3', '\0', '\0', '\0', 'k', 'e',
                            'y',  '\5', '\0', '\0', '\0', 'v',  'a',  'l',  'u', 'e'};

  EXPECT_EQ(ArrowSchemaSetMetadata(&schema, simple_metadata), NANOARROW_OK);
  EXPECT_EQ(memcmp(schema.metadata, simple_metadata, sizeof(simple_metadata)), 0);

  EXPECT_EQ(ArrowSchemaSetMetadata(&schema, nullptr), NANOARROW_OK);
  EXPECT_EQ(schema.metadata, nullptr);

  schema.release(&schema);
}

TEST(SchemaTest, SchemaAllocateDictionary) {
  struct ArrowSchema schema;
  ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED);

  EXPECT_EQ(ArrowSchemaAllocateDictionary(&schema), NANOARROW_OK);
  EXPECT_EQ(schema.dictionary->release, nullptr);
  EXPECT_EQ(ArrowSchemaAllocateDictionary(&schema), EEXIST);
  schema.release(&schema);
}

TEST(SchemaTest, SchemaCopySimpleType) {
  struct ArrowSchema schema;
  ARROW_EXPECT_OK(ExportType(*int32(), &schema));

  struct ArrowSchema schema_copy;
  ArrowSchemaDeepCopy(&schema, &schema_copy);

  ASSERT_NE(schema_copy.release, nullptr);
  EXPECT_STREQ(schema.format, "i");

  schema.release(&schema);
  schema_copy.release(&schema_copy);
}

TEST(SchemaTest, SchemaCopyNestedType) {
  struct ArrowSchema schema;
  auto struct_type = struct_({field("col1", int32())});
  ARROW_EXPECT_OK(ExportType(*struct_type, &schema));

  struct ArrowSchema schema_copy;
  ArrowSchemaDeepCopy(&schema, &schema_copy);

  ASSERT_NE(schema_copy.release, nullptr);
  EXPECT_STREQ(schema_copy.format, "+s");
  EXPECT_EQ(schema_copy.n_children, 1);
  EXPECT_STREQ(schema_copy.children[0]->format, "i");
  EXPECT_STREQ(schema_copy.children[0]->name, "col1");

  schema.release(&schema);
  schema_copy.release(&schema_copy);
}

TEST(SchemaTest, SchemaCopyDictType) {
  struct ArrowSchema schema;
  auto struct_type = dictionary(int32(), int64());
  ARROW_EXPECT_OK(ExportType(*struct_type, &schema));

  struct ArrowSchema schema_copy;
  ArrowSchemaDeepCopy(&schema, &schema_copy);

  ASSERT_STREQ(schema_copy.format, "i");
  ASSERT_NE(schema_copy.dictionary, nullptr);
  EXPECT_STREQ(schema_copy.dictionary->format, "l");

  schema.release(&schema);
  schema_copy.release(&schema_copy);
}

TEST(SchemaTest, SchemaCopyMetadata) {
  struct ArrowSchema schema;
  auto arrow_meta = std::make_shared<KeyValueMetadata>();
  arrow_meta->Append("some_key", "some_value");

  auto int_field = field("field_name", int32(), arrow_meta);
  ARROW_EXPECT_OK(ExportField(*int_field, &schema));

  struct ArrowSchema schema_copy;
  ArrowSchemaDeepCopy(&schema, &schema_copy);

  ASSERT_NE(schema_copy.release, nullptr);
  EXPECT_STREQ(schema_copy.name, "field_name");
  EXPECT_NE(schema_copy.metadata, nullptr);

  auto int_field_roundtrip = ImportField(&schema_copy).ValueOrDie();
  EXPECT_EQ(int_field->name(), int_field_roundtrip->name());
  EXPECT_EQ(int_field_roundtrip->metadata()->Get("some_key").ValueOrDie(), "some_value");

  schema.release(&schema);
}
