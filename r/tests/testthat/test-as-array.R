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

test_that("as_nanoarrow_array() works for nanoarrow_array", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(as_nanoarrow_array(array), array)

  skip_if_not_installed("arrow")
  casted <- as_nanoarrow_array(array, schema = na_int64())
  expect_identical(infer_nanoarrow_schema(casted)$format, "l")
  expect_identical(convert_array(casted), as.double(1:10))
})

test_that("as_nanoarrow_array() works for logical() -> bool()", {
  # Without nulls
  array <- as_nanoarrow_array(c(TRUE, FALSE, TRUE, FALSE), schema = na_bool())
  expect_identical(infer_nanoarrow_schema(array)$format, "b")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(packBits(c(TRUE, FALSE, TRUE, FALSE, rep(FALSE, 4))))
  )

  # With nulls
  array <- as_nanoarrow_array(c(TRUE, FALSE, NA), schema = na_bool())
  expect_identical(infer_nanoarrow_schema(array)$format, "b")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 2), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(packBits(c(TRUE, FALSE, FALSE, rep(FALSE, 5))))
  )
})

test_that("as_nanoarrow_array() works for logical() -> na_int32()", {
  # Without nulls
  array <- as_nanoarrow_array(c(TRUE, FALSE, TRUE, FALSE), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(TRUE, FALSE, TRUE, FALSE)))
  )

  # With nulls
  array <- as_nanoarrow_array(c(TRUE, FALSE, NA), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 2), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(TRUE, FALSE, NA)))
  )
})

test_that("as_nanoarrow_array() works for integer() -> na_int32()", {
  # Without nulls
  array <- as_nanoarrow_array(1:10)
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(as.raw(array$buffers[[2]]), as.raw(as_nanoarrow_buffer(1:10)))

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA))
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(1:10, NA)))
  )
})

test_that("as_nanoarrow_array() works for integer -> na_int64()", {
  skip_if_not_installed("arrow")
  casted <- as_nanoarrow_array(1:10, schema = na_int64())
  expect_identical(infer_nanoarrow_schema(casted)$format, "l")
  expect_identical(convert_array(casted), as.double(1:10))
})

test_that("as_nanoarrow_array() works for double() -> na_double()", {
  # Without nulls
  array <- as_nanoarrow_array(as.double(1:10))
  expect_identical(infer_nanoarrow_schema(array)$format, "g")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(as.double(1:10)))
  )

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA_real_))
  expect_identical(infer_nanoarrow_schema(array)$format, "g")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(1:10, NA_real_)))
  )
})

test_that("as_nanoarrow_array() works for double -> na_int64()", {
  skip_if_not_installed("arrow")
  casted <- as_nanoarrow_array(as.double(1:10), schema = na_int64())
  expect_identical(infer_nanoarrow_schema(casted)$format, "l")
  expect_identical(convert_array(casted), as.double(1:10))
})

test_that("as_nanoarrow_array() works for character() -> na_string()", {
  # Without nulls
  array <- as_nanoarrow_array(letters)
  expect_identical(infer_nanoarrow_schema(array)$format, "u")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(0:26))
  )
  expect_identical(
    as.raw(array$buffers[[3]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )

  # With nulls
  array <- as_nanoarrow_array(c(letters, NA))
  expect_identical(infer_nanoarrow_schema(array)$format, "u")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 26), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(0:26, 26L)))
  )
  expect_identical(
    as.raw(array$buffers[[3]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )
})

test_that("as_nanoarrow_array() works for character() -> na_large_string()", {
  skip_if_not_installed("arrow")

  # Without nulls
  array <- as_nanoarrow_array(letters, schema = na_large_string())
  expect_identical(infer_nanoarrow_schema(array)$format, "U")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[3]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )

  # With nulls
  array <- as_nanoarrow_array(c(letters, NA), schema = na_large_string())
  expect_identical(infer_nanoarrow_schema(array)$format, "U")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 26), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[3]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )
})

test_that("as_nanoarrow_array() works for data.frame() -> na_struct()", {
  array <- as_nanoarrow_array(data.frame(x = 1:10))
  expect_identical(array$length, 10L)
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(infer_nanoarrow_schema(array)$format, "+s")
  expect_identical(infer_nanoarrow_schema(array$children$x)$format, "i")
  expect_identical(as.raw(array$children$x$buffers[[2]]), as.raw(as_nanoarrow_buffer(1:10)))
})
