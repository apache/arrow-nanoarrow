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

test_that("nanoarrow_pointer_is_valid() works", {
  expect_true(nanoarrow_pointer_is_valid(infer_nanoarrow_schema(integer())))
  expect_true(nanoarrow_pointer_is_valid(as_nanoarrow_array(integer())))
  expect_true(nanoarrow_pointer_is_valid(
    as_nanoarrow_array_stream(data.frame(a = integer())))
  )

  expect_false(nanoarrow_pointer_is_valid(nanoarrow_allocate_schema()))
  expect_false(nanoarrow_pointer_is_valid(nanoarrow_allocate_array()))
  expect_false(nanoarrow_pointer_is_valid(nanoarrow_allocate_array_stream()))

  expect_error(nanoarrow_pointer_is_valid(NULL), "must inherit from")
})

test_that("nanoarrow_pointer_release() works", {
  ptr <- infer_nanoarrow_schema(integer())
  expect_true(nanoarrow_pointer_is_valid(ptr))
  nanoarrow_pointer_release(ptr)
  expect_false(nanoarrow_pointer_is_valid(ptr))

  ptr <- as_nanoarrow_array(integer())
  expect_true(nanoarrow_pointer_is_valid(ptr))
  nanoarrow_pointer_release(ptr)
  expect_false(nanoarrow_pointer_is_valid(ptr))

  ptr <- as_nanoarrow_array_stream(data.frame(a = integer()))
  expect_true(nanoarrow_pointer_is_valid(ptr))
  nanoarrow_pointer_release(ptr)
  expect_false(nanoarrow_pointer_is_valid(ptr))

  expect_error(nanoarrow_pointer_release(NULL), "must inherit from")
})

test_that("nanoarrow_pointer_move() works for schema", {
  ptr <- infer_nanoarrow_schema(integer())
  dst <- nanoarrow_allocate_schema()
  nanoarrow_pointer_move(ptr, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(arrow::as_data_type(dst)$Equals(arrow::int32()))

  expect_error(
    nanoarrow_pointer_move(ptr, dst),
    "`ptr_dst` is a valid struct ArrowSchema"
  )

  expect_error(
    nanoarrow_pointer_move(nanoarrow_allocate_schema(), ptr),
    "`ptr_src` is not a valid struct ArrowSchema"
  )
})

test_that("nanoarrow_pointer_move() works for array", {
  ptr <- as_nanoarrow_array(integer())
  dst <- nanoarrow_allocate_array()
  nanoarrow_pointer_move(ptr, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(arrow::as_arrow_array(dst)$Equals(arrow::Array$create(integer())))

  expect_error(
    nanoarrow_pointer_move(ptr, dst),
    "`ptr_dst` is a valid struct ArrowArray"
  )

  expect_error(
    nanoarrow_pointer_move(nanoarrow_allocate_array(), ptr),
    "`ptr_src` is not a valid struct ArrowArray"
  )
})

test_that("nanoarrow_pointer_move() works for array_stream", {
  ptr <- as_nanoarrow_array_stream(data.frame(a = integer()))
  dst <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_move(ptr, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(nanoarrow_pointer_is_valid(dst))

  expect_error(
    nanoarrow_pointer_move(ptr, dst),
    "`ptr_dst` is a valid struct ArrowArrayStream"
  )

  expect_error(
    nanoarrow_pointer_move(nanoarrow_allocate_array_stream(), ptr),
    "`ptr_src` is not a valid struct ArrowArrayStream"
  )
})

test_that("nanoarrow_pointer_move() can import from chr address", {
  ptr <- infer_nanoarrow_schema(integer())
  ptr_chr <- nanoarrow_pointer_addr_chr(ptr)
  dst <- nanoarrow_allocate_schema()

  nanoarrow_pointer_move(ptr_chr, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(nanoarrow_pointer_is_valid(dst))
})

test_that("nanoarrow_pointer_move() can import from dbl address", {
  ptr <- infer_nanoarrow_schema(integer())
  ptr_dbl <- nanoarrow_pointer_addr_dbl(ptr)
  dst <- nanoarrow_allocate_schema()

  nanoarrow_pointer_move(ptr_dbl, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(nanoarrow_pointer_is_valid(dst))
})

test_that("nanoarrow_pointer_move() errors for bad input", {
  ptr <- infer_nanoarrow_schema(integer())
  dst <- nanoarrow_allocate_schema()
  expect_error(nanoarrow_pointer_move(ptr, NULL), "`ptr_dst` must inherit from")
  expect_error(
    nanoarrow_pointer_move(NULL, dst),
    "Pointer must be chr\\[1\\], dbl\\[1\\], or external pointer"
  )
})

test_that("nanoarrow_pointer_export() works for schema", {
  ptr <- infer_nanoarrow_schema(integer())
  dst <- nanoarrow_allocate_schema()
  nanoarrow_pointer_export(ptr, dst)
  expect_true(nanoarrow_pointer_is_valid(ptr))
  expect_true(
    arrow::as_data_type(dst)$Equals(arrow::int32())
  )

  expect_error(
    nanoarrow_pointer_export(ptr, dst),
    "`ptr_dst` is a valid struct ArrowSchema"
  )

  expect_error(
    nanoarrow_pointer_export(nanoarrow_allocate_schema(), nanoarrow_allocate_schema()),
    "has already been released"
  )
})

test_that("nanoarrow_pointer_export() works for array", {
  ptr <- as_nanoarrow_array(integer())
  dst <- nanoarrow_allocate_array()
  nanoarrow_pointer_export(ptr, dst)
  expect_true(nanoarrow_pointer_is_valid(ptr))
  # (when exporting the schema is not included)
  nanoarrow_array_set_schema(dst, infer_nanoarrow_schema(ptr))
  expect_true(arrow::as_arrow_array(dst)$Equals(arrow::Array$create(integer())))

  expect_error(
    nanoarrow_pointer_export(ptr, dst),
    "`ptr_dst` is a valid struct ArrowArray"
  )

  expect_error(
    nanoarrow_pointer_export(nanoarrow_allocate_array(), nanoarrow_allocate_array()),
    "has already been released"
  )
})

test_that("nanoarrow_pointer_export() works for array_stream", {
  ptr <- as_nanoarrow_array_stream(data.frame(a = integer()))
  dst <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_export(ptr, dst)
  expect_false(nanoarrow_pointer_is_valid(ptr))
  expect_true(nanoarrow_pointer_is_valid(dst))

  expect_error(
    nanoarrow_pointer_export(ptr, dst),
    "`ptr_dst` is a valid struct ArrowArrayStream"
  )

  expect_error(
    nanoarrow_pointer_export(nanoarrow_allocate_array_stream(), ptr),
    "is not a valid struct ArrowArrayStream"
  )
})

test_that("nanoarrow_pointer_export() errors for unknown object", {
  expect_error(nanoarrow_pointer_export(NULL), "must inherit from")
})

test_that("pointer address getters work", {
  schema <- infer_nanoarrow_schema(integer())
  expect_match(nanoarrow_pointer_addr_chr(schema), "^[0-9]+$")
  expect_match(nanoarrow_pointer_addr_pretty(schema), "^(0x)?[0-9a-fA-F]+$")
})
