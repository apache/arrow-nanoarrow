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

  array <- as_nanoarrow_array(1:10, schema = na_int32())
  expect_identical(as_nanoarrow_array(array), array)

  skip_if_not_installed("arrow")
  casted <- as_nanoarrow_array(array, schema = na_int64())
  expect_identical(infer_nanoarrow_schema(casted)$format, "l")
  expect_identical(convert_array(casted), as.double(1:10))
})

test_that("as_nanoarrow_array() works for logical() -> na_bool()", {
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

test_that("as_nanoarrow_array() errors for bad logical() creation", {
  skip_if_not_installed("arrow")
  expect_snapshot_error(
    as_nanoarrow_array(TRUE, schema = na_string())
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

test_that("as_nanoarrow_array() works for double() -> na_int32()", {
  # Without nulls
  array <- as_nanoarrow_array(as.double(1:10), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(1:10))
  )

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA_real_), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  # The last element here is (int)NaN not NA_integer_
  expect_identical(
    head(as.raw(array$buffers[[2]]), 10 * 4L),
    as.raw(as_nanoarrow_buffer(1:10))
  )

  # With overflow
  expect_warning(
    as_nanoarrow_array(.Machine$integer.max + as.double(1:5), schema = na_int32()),
    "5 value\\(s\\) overflowed"
  )
})

test_that("as_nanoarrow_array() works for double() -> na_int64()", {
  # Without nulls
  array <- as_nanoarrow_array(as.double(1:10), schema = na_int64())
  expect_identical(infer_nanoarrow_schema(array)$format, "l")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  # This *is* how we create int64 buffers, so just check the roundtrip
  expect_identical(convert_array(array), as.double(1:10))

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA_real_), schema = na_int64())
  expect_identical(infer_nanoarrow_schema(array)$format, "l")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(convert_array(array), as.double(c(1:10, NA_real_)))
})

test_that("as_nanoarrow_array() works for double -> na_int8()", {
  skip_if_not_installed("arrow")
  casted <- as_nanoarrow_array(as.double(1:10), schema = na_int8())
  expect_identical(infer_nanoarrow_schema(casted)$format, "c")
  expect_identical(convert_array(casted), 1:10)
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

test_that("as_nanoarrow_array() works for factor() -> na_dictionary()", {
  array <- as_nanoarrow_array(
    factor(letters),
    schema = na_dictionary(na_string(), na_int32())
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(infer_nanoarrow_schema(array$dictionary)$format, "u")

  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(0:25))
  )

  expect_identical(
    as.raw(array$dictionary$buffers[[3]]),
    charToRaw(paste0(letters, collapse = ""))
  )
})

test_that("as_nanoarrow_array() works for factor() -> na_string()", {
  array <- as_nanoarrow_array(
    factor(letters),
    schema = na_string()
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "u")
  expect_null(array$dictionary)

  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(0:26))
  )
  expect_identical(
    as.raw(array$buffers[[3]]),
    charToRaw(paste0(letters, collapse = ""))
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

test_that("as_nanoarrow_array() errors for bad data.frame() -> na_struct()", {
  expect_error(
    as_nanoarrow_array(data.frame(x = 1:10), schema = na_struct()),
    "Expected 1 schema children"
  )

  skip_if_not_installed("arrow")
  expect_snapshot_error(
    as_nanoarrow_array(data.frame(x = 1:10), schema = na_int32())
  )
})

test_that("as_nanoarrow_array() works for blob::blob() -> na_binary()", {
  skip_if_not_installed("blob")

  # Without nulls
  array <- as_nanoarrow_array(blob::as_blob(letters))
  expect_identical(infer_nanoarrow_schema(array)$format, "z")
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
  array <- as_nanoarrow_array(blob::as_blob(c(letters, NA)))
  expect_identical(infer_nanoarrow_schema(array)$format, "z")
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

test_that("as_nanoarrow_array() works for blob::blob() -> na_large_binary()", {
  skip_if_not_installed("arrow")

  # Without nulls
  array <- as_nanoarrow_array(blob::as_blob(letters), schema = na_large_binary())
  expect_identical(infer_nanoarrow_schema(array)$format, "Z")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[3]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )

  # With nulls
  array <- as_nanoarrow_array(
    blob::as_blob(c(letters, NA)),
    schema = na_large_binary()
  )
  expect_identical(infer_nanoarrow_schema(array)$format, "Z")
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

test_that("as_nanoarrow_array() works for unspecified() -> na_na()", {
  skip_if_not_installed("vctrs")

  array <- as_nanoarrow_array(vctrs::unspecified(5))
  expect_identical(infer_nanoarrow_schema(array)$format, "n")
  expect_identical(array$length, 5L)
  expect_identical(array$null_count, 5L)
})

test_that("as_nanoarrow_array() works for bad unspecified() create", {
  skip_if_not_installed("vctrs")
  skip_if_not_installed("arrow")
  expect_snapshot_error(
    as_nanoarrow_array(vctrs::unspecified(5), schema = na_interval_day_time())
  )
})
