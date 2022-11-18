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

test_that("nanoarrow_schema format, print, and str methods work", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_identical(format(schema), "<nanoarrow_schema int32>")
  expect_output(expect_identical(str(schema), schema), "nanoarrow_schema")
  expect_output(expect_identical(print(schema), schema), "nanoarrow_schema")
})

test_that("nanoarrow_schema format, print, and str methods work for invalid pointers", {
  schema <- nanoarrow_allocate_schema()
  expect_identical(format(schema), "<nanoarrow_schema [invalid: schema is released]>")
  expect_output(expect_identical(str(schema), schema), "nanoarrow_schema")
  expect_output(expect_identical(print(schema), schema), "nanoarrow_schema")
})

test_that("as_nanoarrow_schema() works for nanoarrow_schema", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_identical(as_nanoarrow_schema(schema), schema)
})

test_that("infer_nanoarrow_schema() default method works", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))
})

test_that("nanoarrow_schema_parse() works", {
  simple_info <- nanoarrow_schema_parse(arrow::int32())
  expect_identical(simple_info$type, "int32")
  expect_identical(simple_info$storage_type, "int32")

  fixed_size_info <- nanoarrow_schema_parse(arrow::fixed_size_binary(1234))
  expect_identical(fixed_size_info$fixed_size, 1234L)

  decimal_info <- nanoarrow_schema_parse(arrow::decimal128(4, 5))
  expect_identical(decimal_info$decimal_bitwidth, 128L)
  expect_identical(decimal_info$decimal_precision, 4L)
  expect_identical(decimal_info$decimal_scale, 5L)

  time_unit_info <- nanoarrow_schema_parse(arrow::time32("s"))
  expect_identical(time_unit_info$time_unit, "s")

  timezone_info <- nanoarrow_schema_parse(arrow::timestamp("s", "America/Halifax"))
  expect_identical(timezone_info$timezone, "America/Halifax")

  recursive_info <- nanoarrow_schema_parse(
    infer_nanoarrow_schema(data.frame(x = 1L)),
    recursive = FALSE
  )
  expect_null(recursive_info$children)

  recursive_info <- nanoarrow_schema_parse(
    infer_nanoarrow_schema(data.frame(x = 1L)),
    recursive = TRUE
  )
  expect_length(recursive_info$children, 1L)
  expect_identical(
    recursive_info$children$x,
    nanoarrow_schema_parse(infer_nanoarrow_schema(1L))
  )
})

test_that("schema list interface works for non-nested types", {
  schema <- infer_nanoarrow_schema(1:10)
  expect_identical(length(schema), 6L)
  expect_identical(
    names(schema),
    c("format", "name", "metadata", "flags", "children", "dictionary")
  )
  expect_identical(schema$format, "i")
  expect_identical(schema$name, "")
  expect_identical(schema$metadata, list())
  expect_identical(schema$flags, 2L)
  expect_identical(schema$children, NULL)
  expect_identical(schema$dictionary, NULL)
})

test_that("schema list interface works for nested types", {
  schema <- infer_nanoarrow_schema(data.frame(a = 1L, b = "two"))

  expect_identical(schema$format, "+s")
  expect_named(schema$children, c("a", "b"))
  expect_identical(schema$children$a, schema$children[[1]])
  expect_identical(schema$children$a$format, "i")
  expect_identical(schema$children$b$format, "u")
  expect_s3_class(schema$children$a, "nanoarrow_schema")
  expect_s3_class(schema$children$b, "nanoarrow_schema")

  info_recursive <- nanoarrow_schema_proxy(schema, recursive = TRUE)
  expect_type(info_recursive$children$a, "list")
  expect_identical(info_recursive$children$a$format, "i")
})

test_that("schema list interface works for dictionary types", {
  schema <- infer_nanoarrow_schema(factor(letters[1:5]))

  expect_identical(schema$format, "c")
  expect_identical(schema$dictionary$format, "u")
  expect_s3_class(schema$dictionary, "nanoarrow_schema")

  info_recursive <- nanoarrow_schema_proxy(schema, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_identical(info_recursive$dictionary$format, "u")
})

test_that("schema list interface works with metadata", {
  schema <- infer_nanoarrow_schema(as.POSIXlt("2020-01-01", tz = "UTC"))
  expect_identical(
    schema$metadata[["ARROW:extension:name"]],
    "arrow.r.vctrs"
  )
  expect_s3_class(
    unserialize(schema$metadata[["ARROW:extension:metadata"]]),
    "POSIXlt"
  )
})
