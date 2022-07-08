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

TEST(SchemaViewTest, SchemaViewInitErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, nullptr, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error), "Expected non-NULL schema");

  schema.release = nullptr;
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error), "Expected non-released schema");

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_UNINITIALIZED), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Error parsing schema->format: Expected a null-terminated string but found NULL");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, ""), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected a string with size > 0");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Unknown format: '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "n*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format 'n*': parsed 1/2 characters");

  schema.release(&schema);
}

void ExpectSimpleTypeOk(std::shared_ptr<DataType> arrow_t, enum ArrowType nanoarrow_t) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*arrow_t, &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, nanoarrow_t);
  EXPECT_EQ(schema_view.storage_data_type, nanoarrow_t);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitSimple) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*null(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_NA);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_NA);
  EXPECT_EQ(schema_view.n_buffers, 0);
  EXPECT_EQ(schema_view.extension_name.data, nullptr);
  EXPECT_EQ(schema_view.extension_metadata.data, nullptr);
  schema.release(&schema);

  ExpectSimpleTypeOk(boolean(), NANOARROW_TYPE_BOOL);
  ExpectSimpleTypeOk(int8(), NANOARROW_TYPE_INT8);
  ExpectSimpleTypeOk(uint8(), NANOARROW_TYPE_UINT8);
  ExpectSimpleTypeOk(int16(), NANOARROW_TYPE_INT16);
  ExpectSimpleTypeOk(uint16(), NANOARROW_TYPE_UINT16);
  ExpectSimpleTypeOk(int32(), NANOARROW_TYPE_INT32);
  ExpectSimpleTypeOk(uint32(), NANOARROW_TYPE_UINT32);
  ExpectSimpleTypeOk(int64(), NANOARROW_TYPE_INT64);
  ExpectSimpleTypeOk(uint64(), NANOARROW_TYPE_UINT64);
  ExpectSimpleTypeOk(float16(), NANOARROW_TYPE_HALF_FLOAT);
  ExpectSimpleTypeOk(float64(), NANOARROW_TYPE_DOUBLE);
  ExpectSimpleTypeOk(float32(), NANOARROW_TYPE_FLOAT);
}

TEST(SchemaViewTest, SchemaViewInitSimpleErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected schema with 0 children but found 2 children");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitDecimal) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*decimal128(5, 6), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DECIMAL128);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_DECIMAL128);
  EXPECT_EQ(schema_view.decimal_bitwidth, 128);
  EXPECT_EQ(schema_view.decimal_precision, 5);
  EXPECT_EQ(schema_view.decimal_scale, 6);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*decimal256(5, 6), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DECIMAL256);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_DECIMAL256);
  EXPECT_EQ(schema_view.decimal_bitwidth, 256);
  EXPECT_EQ(schema_view.decimal_precision, 5);
  EXPECT_EQ(schema_view.decimal_scale, 6);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitDecimalErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':precision,scale[,bitwidth]' "
               "following 'd'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d:"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':precision,scale[,bitwidth]' "
               "following 'd'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d:5"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 'precision,scale[,bitwidth]' "
               "following 'd:'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d:5,"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 'scale[,bitwidth]' following "
               "'d:precision,'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d:5,6,"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Error parsing schema->format: Expected precision following 'd:precision,scale,'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "d:5,6,127"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected decimal bitwidth of 128 or 256 "
               "but found 127");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitBinaryAndString) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*fixed_size_binary(123), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_FIXED_SIZE_BINARY);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_FIXED_SIZE_BINARY);
  EXPECT_EQ(schema_view.fixed_size, 123);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*utf8(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 3);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_buffer_id, 2);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_STRING);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_STRING);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*binary(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 3);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_buffer_id, 2);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_BINARY);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_BINARY);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*large_binary(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 3);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_buffer_id, 2);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_LARGE_BINARY);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_LARGE_BINARY);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*large_utf8(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 3);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_buffer_id, 2);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_LARGE_STRING);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_LARGE_STRING);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitBinaryAndStringErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "w"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':<width>' following 'w'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "w:"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':<width>' following 'w'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "w:abc"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format 'w:abc': parsed 2/5 characters");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "w:0"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected size > 0 for fixed size binary but found size 0");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeDate) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*date32(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DATE32);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*date64(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DATE64);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeTime) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*time32(TimeUnit::SECOND), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIME32);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_SECOND);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*time32(TimeUnit::MILLI), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIME32);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MILLI);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*time64(TimeUnit::MICRO), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIME64);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MICRO);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*time64(TimeUnit::NANO), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIME64);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_NANO);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeTimestamp) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*timestamp(TimeUnit::SECOND, "America/Halifax"), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIMESTAMP);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_SECOND);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*timestamp(TimeUnit::MILLI, "America/Halifax"), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIMESTAMP);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MILLI);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*timestamp(TimeUnit::MICRO, "America/Halifax"), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIMESTAMP);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MICRO);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*timestamp(TimeUnit::NANO, "America/Halifax"), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_TIMESTAMP);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_NANO);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeDuration) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*duration(TimeUnit::SECOND), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DURATION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_SECOND);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*duration(TimeUnit::MILLI), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DURATION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MILLI);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*duration(TimeUnit::MICRO), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DURATION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_MICRO);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*duration(TimeUnit::NANO), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DURATION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT64);
  EXPECT_EQ(schema_view.time_unit, NANOARROW_TIME_UNIT_NANO);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeInterval) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*month_interval(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_INTERVAL_MONTHS);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INTERVAL_MONTHS);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*day_time_interval(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_INTERVAL_DAY_TIME);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INTERVAL_DAY_TIME);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*month_day_nano_interval(), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitTimeErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "t*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 'd', 't', 's', 'D', or 'i' "
               "following 't' but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "td*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Error parsing schema->format: Expected 'D' or 'm' following 'td' but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "tt*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 's', 'm', 'u', or 'n' following "
               "'tt' but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "ts*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 's', 'm', 'u', or 'n' following "
               "'ts' but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "tD*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 's', 'm', u', or 'n' following "
               "'tD' but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "ti*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected 'M', 'D', or 'n' following 'ti' "
               "but found '*'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "tss"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':' following 'tss' but found ''");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedList) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*list(int32()), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_LIST);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_LIST);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*large_list(int32()), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_LARGE_LIST);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_LARGE_LIST);
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*fixed_size_list(int32(), 123), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 1);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_FIXED_SIZE_LIST);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_FIXED_SIZE_LIST);
  EXPECT_EQ(schema_view.fixed_size, 123);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewNestedListErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+w"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':<width>' following '+w'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+w:"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected ':<width>' following '+w'");

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+w:1"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected schema with 1 children but found 0 children");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedStruct) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*struct_({field("col", int32())}), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 1);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_STRUCT);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_STRUCT);

  // Make sure child validates
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, schema.children[0], &error), NANOARROW_OK);

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedStructErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Expected valid schema at schema->children[0] but found a released schema");

  // Make sure validation passes even with an inspectable but invalid child
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_UNINITIALIZED),
            NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, schema.children[0], &error), EINVAL);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);

  ArrowFree(schema.children[0]);
  schema.children[0] = NULL;
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected valid schema at schema->children[0] but found NULL");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedMap) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*map(int32(), int32()), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 1);
  EXPECT_EQ(schema_view.validity_buffer_id, 0);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_MAP);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_MAP);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedMapErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_MAP), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 2), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected schema with 1 children but found 2 children");
  schema.release(&schema);

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_MAP), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_UNINITIALIZED),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetFormat(schema.children[0], "n"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected child of map type to have 2 children but found 0");
  schema.release(&schema);

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_MAP), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(&schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0], NANOARROW_TYPE_UNINITIALIZED),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema.children[0], 2), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetFormat(schema.children[0], "+us:0,1"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0]->children[0], NANOARROW_TYPE_NA),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.children[0]->children[1], NANOARROW_TYPE_NA),
            NANOARROW_OK);

  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Expected format of child of map type to be '+s' but found '+us:0,1'");
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedUnion) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*dense_union({field("col", int32())}), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 2);
  EXPECT_EQ(schema_view.type_id_buffer_id, 0);
  EXPECT_EQ(schema_view.offset_buffer_id, 1);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DENSE_UNION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_DENSE_UNION);
  EXPECT_EQ(
      std::string(schema_view.union_type_ids.data, schema_view.union_type_ids.n_bytes),
      std::string("0"));
  schema.release(&schema);

  ARROW_EXPECT_OK(ExportType(*sparse_union({field("col", int32())}), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.n_buffers, 1);
  EXPECT_EQ(schema_view.type_id_buffer_id, 0);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_SPARSE_UNION);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_SPARSE_UNION);
  EXPECT_EQ(
      std::string(schema_view.union_type_ids.data, schema_view.union_type_ids.n_bytes),
      std::string("0"));
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitNestedUnionErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_NA), NANOARROW_OK);

  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+u*"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected union format string "
               "+us:<type_ids> or +ud:<type_ids> but found '+u*'");

  // missing colon
  ASSERT_EQ(ArrowSchemaSetFormat(&schema, "+us"), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error),
               "Error parsing schema->format: Expected union format string "
               "+us:<type_ids> or +ud:<type_ids> but found '+us'");

  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitDictionary) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ARROW_EXPECT_OK(ExportType(*dictionary(int32(), utf8()), &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(schema_view.storage_data_type, NANOARROW_TYPE_INT32);
  EXPECT_EQ(schema_view.data_type, NANOARROW_TYPE_DICTIONARY);
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitDictionaryErrors) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(&schema), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error), "Expected non-released schema");
  schema.release(&schema);

  ASSERT_EQ(ArrowSchemaInit(&schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(&schema), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInit(schema.dictionary, NANOARROW_TYPE_STRING), NANOARROW_OK);
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), EINVAL);
  EXPECT_STREQ(
      ArrowErrorMessage(&error),
      "Expected dictionary schema index type to be an integral type but found '+s'");
  schema.release(&schema);
}

TEST(SchemaViewTest, SchemaViewInitExtension) {
  struct ArrowSchema schema;
  struct ArrowSchemaView schema_view;
  struct ArrowError error;

  auto arrow_meta = std::make_shared<KeyValueMetadata>();
  arrow_meta->Append("ARROW:extension:name", "arrow.test.ext_name");
  arrow_meta->Append("ARROW:extension:metadata", "test metadata");

  auto int_field = field("field_name", int32(), arrow_meta);
  ARROW_EXPECT_OK(ExportField(*int_field, &schema));
  EXPECT_EQ(ArrowSchemaViewInit(&schema_view, &schema, &error), NANOARROW_OK);
  EXPECT_EQ(
      std::string(schema_view.extension_name.data, schema_view.extension_name.n_bytes),
      "arrow.test.ext_name");
  EXPECT_EQ(std::string(schema_view.extension_metadata.data,
                        schema_view.extension_metadata.n_bytes),
            "test metadata");

  schema.release(&schema);
}
