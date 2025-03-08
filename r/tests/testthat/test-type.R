# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

test_that("type constructors for parameter-free types work", {
  # Some of these types have parameters but also have default values
  parameter_free_types <- c(
    "na", "bool", "uint8", "int8", "uint16", "int16",
    "uint32", "int32", "uint64", "int64", "half_float", "float",
    "double", "string", "binary", "date32",
    "date64", "timestamp", "time32", "time64", "interval_months",
    "interval_day_time", "struct",
    "duration", "large_string", "large_binary",
    "interval_month_day_nano"
  )

  for (type_name in parameter_free_types) {
    # Check that the right type gets created
    expect_identical(
      nanoarrow_schema_parse(na_type(!!type_name))$type,
      !!type_name
    )

    # Check that the default schema is nullable
    if (type_name == "struct") {
      expect_identical(na_type(!!type_name)$flags, 0L)
    } else {
      expect_identical(na_type(!!type_name)$flags, 2L)
    }

    # Check that non-nullable schemas are non-nullable
    expect_identical(na_type(!!type_name, nullable = FALSE)$flags, 0L)
  }
})

test_that("non-logical nullable values do not crash", {
  expect_identical(na_na(nullable = NULL)$flags, 0L)
  expect_identical(na_time32(nullable = NULL)$flags, 0L)
  expect_identical(na_fixed_size_binary(1, nullable = NULL)$flags, 0L)
  expect_identical(na_decimal128(1, 1, nullable = NULL)$flags, 0L)
})

test_that("timestamp type passes along timezone parameter", {
  schema <- na_timestamp(timezone = "UTC")
  expect_identical(nanoarrow_schema_parse(schema)$timezone, "UTC")

  expect_error(
    na_timestamp(timezone = NULL),
    "must be character"
  )

  expect_error(
    na_timestamp(timezone = NA_character_),
    "must be character"
  )

  expect_error(
    na_timestamp(timezone = character()),
    "must be character"
  )
})

test_that("decimal types pass along precision and scale", {
  schema <- na_decimal32(12, 10)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_bitwidth, 32L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_precision, 12L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_scale, 10L)

  schema <- na_decimal64(12, 10)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_bitwidth, 64L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_precision, 12L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_scale, 10L)

  schema <- na_decimal128(12, 10)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_bitwidth, 128L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_precision, 12L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_scale, 10L)

  schema <- na_decimal256(12, 10)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_precision, 12L)
  expect_identical(nanoarrow_schema_parse(schema)$decimal_scale, 10L)
})

test_that("fixed-size binary passes along fixed-size parameter", {
  schema <- na_fixed_size_binary(123)
  expect_identical(nanoarrow_schema_parse(schema)$fixed_size, 123L)
})

test_that("struct constructor passes along children", {
  schema <- na_struct(list(col_name = na_int32()))
  expect_identical(schema$format, "+s")
  expect_named(schema$children, "col_name")
  expect_identical(schema$children[[1]]$format, "i")
})

test_that("struct constructor passes along children", {
  schema <- na_struct(list(col_name = na_int32()))
  expect_identical(schema$format, "+s")
  expect_named(schema$children, "col_name")
  expect_identical(schema$children[[1]]$format, "i")
})

test_that("sparse and dense unions can be created", {
  schema <- na_sparse_union(list(na_int32(), na_string()))
  expect_identical(nanoarrow_schema_parse(schema)$union_type_ids, c(0L, 1L))

  schema <- na_dense_union(list(na_int32(), na_string()))
  expect_identical(nanoarrow_schema_parse(schema)$union_type_ids, c(0L, 1L))
})

test_that("list constructors assign the correct child type", {
  schema <- na_list(na_int32())
  expect_identical(schema$format, "+l")
  expect_named(schema$children, "item")
  expect_identical(schema$children[[1]]$format, "i")

  schema <- na_large_list(na_int32())
  expect_identical(schema$format, "+L")
  expect_named(schema$children, "item")
  expect_identical(schema$children[[1]]$format, "i")

  schema <- na_fixed_size_list(na_int32(), 123)
  expect_identical(schema$format, "+w:123")
  expect_named(schema$children, "item")
  expect_identical(schema$children[[1]]$format, "i")
})

test_that("list_view constructors assign the correct child type", {
  schema <- na_list_view(na_int32())
  expect_identical(schema$format, "+vl")
  expect_named(schema$children, "item")
  expect_identical(schema$children[[1]]$format, "i")

  schema <- na_large_list_view(na_int32())
  expect_identical(schema$format, "+vL")
  expect_named(schema$children, "item")
  expect_identical(schema$children[[1]]$format, "i")
})

test_that("map constructor assigns the correct key and value types", {
  schema <- na_map(na_int32(nullable = FALSE), na_int64())
  expect_named(schema$children, "entries")
  expect_named(schema$children$entries$children, c("key", "value"))
  expect_identical(schema$children$entries$children$key$format, "i")
  expect_identical(schema$children$entries$children$value$format, "l")
})

test_that("dictionary types can be created", {
  schema <- na_dictionary(na_string(), ordered = FALSE)
  expect_identical(schema$format, "i")
  expect_identical(schema$dictionary$format, "u")
  expect_identical(schema$flags, ARROW_FLAG$NULLABLE)

  schema <- na_dictionary(na_string(), ordered = TRUE)
  expect_identical(
    schema$flags,
    bitwOr(ARROW_FLAG$NULLABLE, ARROW_FLAG$DICTIONARY_ORDERED)
  )
})

test_that("extension types can be created", {
  schema <- na_extension(na_int32(), "ext_name", "ext_meta")
  expect_identical(nanoarrow_schema_parse(schema)$extension_name, "ext_name")
  expect_identical(
    nanoarrow_schema_parse(schema)$extension_metadata,
    charToRaw("ext_meta")
  )
})
