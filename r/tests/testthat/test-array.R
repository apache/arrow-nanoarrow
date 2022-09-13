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

test_that("as_nanoarrow_array() / from_nanoarrow_array() default method works", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(from_nanoarrow_array(array), 1:10)

  array <- as_nanoarrow_array(1:10, schema = arrow::float64())
  expect_identical(from_nanoarrow_array(array), as.double(1:10))
})

test_that("infer_nanoarrow_schema() works for nanoarrow_array", {
  array <- as_nanoarrow_array(1:10)
  schema <- infer_nanoarrow_schema(array)
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))

  nanoarrow_array_set_schema(array, NULL)
  expect_error(infer_nanoarrow_schema(array), "has no associated schema")
})

test_that("nanoarrow_array_set_schema() errors for invalid schema/array", {
  array <- as_nanoarrow_array(integer())
  schema <- infer_nanoarrow_schema(character())
  expect_error(
    nanoarrow_array_set_schema(array, schema),
    "Expected array with 3 buffer\\(s\\) but found 2 buffer\\(s\\)"
  )
})

test_that("as.vector() and as.data.frame() work for array", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(as.vector(array), 1:10)

  struct_array <- as_nanoarrow_array(data.frame(a = 1:10))
  expect_identical(as.data.frame(struct_array), data.frame(a = 1:10))
})

test_that("schemaless array list interface works for non-nested types", {
  array <- as_nanoarrow_array(1:10)
  nanoarrow_array_set_schema(array, NULL)

  expect_identical(length(array), 6L)
  expect_identical(
    names(array),
    c("length",  "null_count", "offset", "buffers", "children",   "dictionary")
  )
  expect_identical(array$length, 10L)
  expect_identical(array$null_count, 0L)
  expect_identical(array$offset, 0L)
  expect_length(array$buffers, 2L)
  expect_s3_class(array$buffers[[1]], "nanoarrow_buffer")
  expect_s3_class(array$buffers[[2]], "nanoarrow_buffer")
  expect_null(array$children)
  expect_null(array$dictionary)
})

test_that("schemaless array list interface works for nested types", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  nanoarrow_array_set_schema(array, NULL)

  expect_length(array$children, 2L)
  expect_length(array$children[[1]]$buffers, 2L)
  expect_length(array$children[[2]]$buffers, 3L)
  expect_s3_class(array$children[[1]], "nanoarrow_array")
  expect_s3_class(array$children[[2]], "nanoarrow_array")

  info_recursive <- nanoarrow_array_info(array, recursive = TRUE)
  expect_type(info_recursive$children[[1]], "list")
  expect_length(info_recursive$children[[1]]$buffers, 2L)
})

test_that("schemaless array list interface works for dictionary types", {
  array <- as_nanoarrow_array(factor(letters[1:5]))
  nanoarrow_array_set_schema(array, NULL)

  expect_length(array$buffers, 2L)
  expect_length(array$dictionary$buffers, 3L)
  expect_s3_class(array$dictionary, "nanoarrow_array")

  info_recursive <- nanoarrow_array_info_safe(array, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_length(info_recursive$dictionary$buffers, 3L)
})

test_that("array list interface works for nested types", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))

  expect_named(array$children, c("a", "b"))
  expect_s3_class(array$children[[1]], "nanoarrow_array")
  expect_s3_class(infer_nanoarrow_schema(array$children[[1]]), "nanoarrow_schema")

  expect_s3_class(array$buffers[[1]], "nanoarrow_buffer_validity")
  expect_s3_class(array$children$a$buffers[[2]], "nanoarrow_buffer_data_int32")
  expect_s3_class(array$children$b$buffers[[2]], "nanoarrow_buffer_data_offset32")

  info_recursive <- nanoarrow_array_info_safe(array, recursive = TRUE)
  expect_type(info_recursive$children, "list")
  expect_s3_class(info_recursive$children$a$buffers[[2]], "nanoarrow_buffer_data_int32")
  expect_s3_class(info_recursive$children$b$buffers[[2]], "nanoarrow_buffer_data_offset32")
})

test_that("array list interface works for dictionary types", {
  array <- as_nanoarrow_array(factor(letters[1:5]))

  expect_s3_class(array$buffers[[2]], "nanoarrow_buffer_data_int8")
  expect_s3_class(array$dictionary$buffers[[2]], "nanoarrow_buffer_data_offset32")

  info_recursive <- nanoarrow_array_info_safe(array, recursive = TRUE)
  expect_type(info_recursive$dictionary, "list")
  expect_s3_class(info_recursive$dictionary$buffers[[2]], "nanoarrow_buffer_data_offset32")
})
