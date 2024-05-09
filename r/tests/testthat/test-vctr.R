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

test_that("as_nanoarrow_vctr() works for basic input", {
  array <- as_nanoarrow_array(c("one", "two"))
  vctr <- as_nanoarrow_vctr(array)
  expect_identical(as.integer(unclass(vctr)), 1:2)
  expect_identical(as_nanoarrow_vctr(vctr), vctr)

  expect_identical(infer_nanoarrow_schema(vctr)$format, "u")
  expect_identical(as_nanoarrow_schema(vctr)$format, "u")
})

test_that("format() works for nanoarrow_vctr", {
  array <- as_nanoarrow_array(c("one", "two"))
  vctr <- as_nanoarrow_vctr(array)
  expect_identical(format(vctr),format(c("one", "two")))
})

test_that("nanoarrow_vctr to stream generates an empty stream for empty slice", {
  vctr <- new_nanoarrow_vctr(list(), na_string())
  stream <- as_nanoarrow_array_stream(vctr)
  schema_out <- stream$get_schema()
  expect_identical(schema_out$format, "u")
  expect_identical(collect_array_stream(stream), list())
})

test_that("nanoarrow_vctr to stream generates identical stream for identity slice", {
  array <- as_nanoarrow_array("one")
  vctr <- new_nanoarrow_vctr(list(array), infer_nanoarrow_schema(array))

  stream <- as_nanoarrow_array_stream(vctr)
  schema_out <- stream$get_schema()
  expect_identical(schema_out$format, "u")

  collected <- collect_array_stream(stream)
  expect_length(collected, 1)
  expect_identical(
    convert_buffer(array$buffers[[3]]),
    "one"
  )
})

test_that("nanoarrow_vctr to stream works for arbitrary slices", {
  array1 <- as_nanoarrow_array(c("one", "two", "three"))
  array2 <- as_nanoarrow_array(c("four", "five", "six", "seven"))
  vctr <- new_nanoarrow_vctr(list(array1, array2), infer_nanoarrow_schema(array1))

  chunks16 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[1:6])
  )
  expect_length(chunks16, 2)
  expect_identical(chunks16[[1]]$offset, 0L)
  expect_identical(chunks16[[1]]$length, 3L)
  expect_identical(chunks16[[2]]$offset, 0L)
  expect_identical(chunks16[[2]]$length, 3L)

  chunks34 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[3:4])
  )
  expect_length(chunks34, 2)
  expect_identical(chunks34[[1]]$offset, 2L)
  expect_identical(chunks34[[1]]$length, 1L)
  expect_identical(chunks34[[2]]$offset, 0L)
  expect_identical(chunks34[[2]]$length, 1L)

  chunks13 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[1:3])
  )
  expect_length(chunks13, 1)
  expect_identical(chunks13[[1]]$offset, 0L)
  expect_identical(chunks13[[1]]$length, 3L)

  chunks46 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[4:6])
  )
  expect_length(chunks46, 1)
  expect_identical(chunks46[[1]]$offset, 0L)
  expect_identical(chunks46[[1]]$length, 3L)

  chunks56 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[5:6])
  )
  expect_length(chunks56, 1)
  expect_identical(chunks56[[1]]$offset, 1L)
  expect_identical(chunks56[[1]]$length, 2L)

  chunks57 <- collect_array_stream(
    as_nanoarrow_array_stream(vctr[5:7])
  )
  expect_length(chunks57, 1)
  expect_identical(chunks57[[1]]$offset, 1L)
  expect_identical(chunks57[[1]]$length, 3L)
})

test_that("Errors occur for unsupported subset operations", {
  array <- as_nanoarrow_array("one")
  vctr <- as_nanoarrow_vctr(array)
  expect_error(
    vctr[5:1],
    "Can't subset nanoarrow_vctr with non-slice"
  )

  expect_error(
    vctr[1] <- "something",
    "subset assignment for nanoarrow_vctr is not supported"
  )

  expect_error(
    vctr[[1]] <- "something",
    "subset assignment for nanoarrow_vctr is not supported"
  )
})

test_that("slice detector works", {
  expect_identical(
    vctr_as_slice(logical()),
    NULL
  )

  expect_identical(
    vctr_as_slice(2:1),
    NULL
  )

  expect_identical(
    vctr_as_slice(integer()),
    c(NA_integer_, 0L)
  )

  expect_identical(
    vctr_as_slice(2L),
    c(2L, 1L)
  )

  expect_identical(
    vctr_as_slice(1:10),
    c(1L, 10L)
  )

  expect_identical(
    vctr_as_slice(10:2048),
    c(10L, (2048L - 10L + 1L))
  )
})

test_that("chunk resolver works", {
  chunk_offset1 <- 0:10

  expect_identical(
    vctr_resolve_chunk(c(-1L, 11L), chunk_offset1),
    c(NA_integer_, NA_integer_)
  )

  expect_identical(
    vctr_resolve_chunk(9:0, chunk_offset1),
    9:0
  )
})
