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
  stream0 <- arrow::RecordBatchReader$create(
    schema = arrow::schema(x = arrow::int32())
  )
  stream0 <- as_nanoarrow_array_stream(stream0)
  expect_identical(convert_array_stream(stream0), data.frame(x = integer()))

  stream1 <- arrow::RecordBatchReader$create(
    arrow::record_batch(x = 1:5)
  )
  stream1 <- as_nanoarrow_array_stream(stream1)
  expect_identical(convert_array_stream(stream1), data.frame(x = 1:5))

  stream2 <- arrow::RecordBatchReader$create(
    arrow::record_batch(x = 1:5),
    arrow::record_batch(x = 6:10)
  )
  stream2 <- as_nanoarrow_array_stream(stream2)
  expect_identical(convert_array_stream(stream2), data.frame(x = 1:10))
})

test_that("convert array stream with explicit size works", {
  stream0 <- arrow::RecordBatchReader$create(
    schema = arrow::schema(x = arrow::int32())
  )
  stream0 <- as_nanoarrow_array_stream(stream0)
  expect_identical(
    convert_array_stream(stream0, size = 0),
    data.frame(x = integer())
  )

  stream1 <- arrow::RecordBatchReader$create(
    arrow::record_batch(x = 1:5)
  )
  stream1 <- as_nanoarrow_array_stream(stream1)
  expect_identical(
    convert_array_stream(stream1, size = 5),
    data.frame(x = 1:5)
  )

  stream2 <- arrow::RecordBatchReader$create(
    arrow::record_batch(x = 1:5),
    arrow::record_batch(x = 6:10)
  )
  stream2 <- as_nanoarrow_array_stream(stream2)
  expect_identical(
    convert_array_stream(stream2, size = 10),
    data.frame(x = 1:10)
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
  raw_posixlt <- as.data.frame(unclass(as.POSIXlt("2021-01-01")))

  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(stream),
    raw_posixlt
  )

  stream <- as_nanoarrow_array_stream(raw_posixlt)
  expect_identical(
    convert_array_stream(stream, as.POSIXlt(character(), tz = "America/Halifax")),
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
      as.POSIXlt(character(), tz = "America/Halifax"),
      size = 1
    ),
    as.POSIXlt("2021-01-01", tz = "America/Halifax")
  )
})
