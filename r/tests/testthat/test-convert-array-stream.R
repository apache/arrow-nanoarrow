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

test_that("convert array stream works", {
  stream0 <- basic_array_stream(list(), schema = na_struct(list(x = na_int32())))
  expect_identical(convert_array_stream(stream0), data.frame(x = integer()))

  stream1 <- basic_array_stream(list(data.frame(x = 1:5)))
  expect_identical(convert_array_stream(stream1), data.frame(x = 1:5))

  stream2 <- basic_array_stream(
    list(
      data.frame(x = 1:5),
      data.frame(x = 6:10)
    )
  )
  expect_identical(convert_array_stream(stream2), data.frame(x = 1:10))

  stream3 <- basic_array_stream(list(), schema = na_int32())
  expect_identical(convert_array_stream(stream3), integer())
})

test_that("convert array stream with explicit size works", {
  stream0 <- basic_array_stream(list(), schema = na_struct(list(x = na_int32())))
  expect_identical(
    convert_array_stream(stream0, size = 0),
    data.frame(x = integer())
  )

  stream1 <- basic_array_stream(list(data.frame(x = 1:5)))
  expect_identical(
    convert_array_stream(stream1, size = 5),
    data.frame(x = 1:5)
  )

  stream2 <- basic_array_stream(
    list(
      data.frame(x = 1:5),
      data.frame(x = 6:10)
    )
  )
  expect_identical(
    convert_array_stream(stream2, size = 10),
    data.frame(x = 1:10)
  )
})

test_that("convert array stream with functional ptype works", {
  tibble_or_bust <- function(array, ptype) {
    if (is.data.frame(ptype)) {
      ptype <- tibble::as_tibble(ptype)
      ptype[] <- Map(tibble_or_bust, list(NULL), ptype)
    }

    ptype
  }

  df_nested_df <- as.data.frame(
    tibble::tibble(a = 1L, b = "two", c = data.frame(a = 3))
  )
  stream_nested <- as_nanoarrow_array_stream(df_nested_df)
  expect_identical(
    convert_array_stream(stream_nested, tibble_or_bust),
    tibble::tibble(a = 1L, b = "two", c = tibble::tibble(a = 3))
  )
})

test_that("convert array stream works for nested data.frames", {
  tbl_nested_df <- tibble::tibble(a = 1L, b = "two", c = data.frame(a = 3))

  stream_nested <- as_nanoarrow_array_stream(tbl_nested_df)
  expect_identical(
    convert_array_stream(stream_nested, tbl_nested_df),
    tbl_nested_df
  )

  stream_nested <- as_nanoarrow_array_stream(tbl_nested_df)
  expect_identical(
    convert_array_stream(stream_nested, size = 1L),
    as.data.frame(tbl_nested_df)
  )

  stream_nested <- as_nanoarrow_array_stream(tbl_nested_df)
  expect_identical(
    convert_array_stream(stream_nested, tbl_nested_df, size = 1L),
    tbl_nested_df
  )
})

test_that("convert array stream works for struct-style vectors", {
  raw_posixlt <- as.data.frame(
    unclass(as.POSIXlt("2021-01-01", tz = "America/Halifax")),
    stringsAsFactors = FALSE
  )

  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(stream),
    raw_posixlt
  )

  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(stream, as.POSIXlt("2021-01-01", tz = "America/Halifax")),
    as.POSIXlt("2021-01-01", tz = "America/Halifax")
  )

  # Check with fixed size since this takes a different code path
  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(stream, size = 1L),
    raw_posixlt
  )

  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(
      stream,
      as.POSIXlt("2021-01-01", tz = "America/Halifax"),
      size = 1
    ),
    as.POSIXlt("2021-01-01", tz = "America/Halifax")
  )
})

test_that("convert array stream works for fixed_size_list_of() -> matrix()", {
  mat <- matrix(1:6, ncol = 2, byrow = TRUE)
  array <- as_nanoarrow_array(mat)
  stream <- basic_array_stream(list(array, array))
  expect_identical(
    convert_array_stream(stream, matrix(integer(), ncol = 2)),
    rbind(mat, mat)
  )

  # Check with non-default ptype
  stream <- basic_array_stream(list(array, array))
  expect_identical(
    convert_array_stream(stream, matrix(double(), ncol = 2)),
    rbind(
      matrix(as.double(1:6), ncol = 2, byrow = TRUE),
      matrix(as.double(1:6), ncol = 2, byrow = TRUE)
    )
  )

  # TODO: Because we special-case the array in the converter, we need
  # - Check the nested case (matrix column in a data frame)
  # - Check the list_of case
  # - Check the "source has non-zero offset" case
})

test_that("convert array stream works for empty fixed_size_list_of() -> matrix()", {
  stream <- basic_array_stream(list(), schema = na_fixed_size_list(na_int32(), 2))
  expect_identical(
    convert_array_stream(stream, matrix(integer(), ncol = 2)),
    matrix(integer(), ncol = 2)
  )
})

test_that("convert array stream works for nested fixed_size_list_of() -> matrix()", {
  df <- data.frame(x = 1:3)
  df$mat <- matrix(1:6, ncol = 2, byrow = TRUE)

  expected <- df[c(1:3, 1:3),]
  row.names(expected) <- 1:6

  array <- as_nanoarrow_array(df)
  stream <- basic_array_stream(list(array, array))
  expect_identical(
    convert_array_stream(stream, df[integer(0),]),
    expected
  )
})

test_that("convert array stream works for fixed_size_list_of() with non-zero offset -> matrix() ", {
  mat <- matrix(1:6, ncol = 2, byrow = TRUE)
  array <- as_nanoarrow_array(mat)
  array <- nanoarrow_array_modify(array, list(offset = 1, length = 2))
  stream <- basic_array_stream(list(array, array))
  expect_identical(
    convert_array_stream(stream, matrix(integer(), ncol = 2)),
    rbind(mat[2:3, ], mat[2:3, ])
  )
})

test_that("convert array stream respects the value of n", {
  batches <- list(
    data.frame(x = 1:5),
    data.frame(x = 6:10),
    data.frame(x = 11:15)
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 0),
    data.frame(x = integer())
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 1),
    data.frame(x = 1:5)
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 2),
    data.frame(x = 1:10)
  )
})

test_that("fixed-size convert array stream respects the value of n", {
  batches <- list(
    data.frame(x = 1:5),
    data.frame(x = 6:10),
    data.frame(x = 11:15)
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 0, size = 0),
    data.frame(x = integer())
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 1, size = 5),
    data.frame(x = 1:5)
  )

  stream3 <- basic_array_stream(batches)
  expect_identical(
    convert_array_stream(stream3, n = 2, size = 10),
    data.frame(x = 1:10)
  )
})

test_that("fixed-size stream conversion errors when the output has insufficient size", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:100))
  expect_error(
    convert_array_stream(stream, size = 2),
    "Expected to materialize 100 values in batch 1 but materialized 2"
  )
})
