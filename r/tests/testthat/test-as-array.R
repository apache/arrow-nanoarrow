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
  casted <- as_nanoarrow_array(1:10, schema = na_int64())
  expect_identical(infer_nanoarrow_schema(casted)$format, "l")
  expect_identical(convert_array(casted), as.double(1:10))
})

test_that("as_nanoarrow_array() works for integer -> na_decimal_xxx()", {
  numbers <- c(1234L, 56L, NA, -10:10)
  schemas <- list(
    na_decimal32(9, 3),
    na_decimal64(9, 3),
    na_decimal128(9, 3),
    na_decimal256(9, 3)
  )

  for (schema in schemas) {
    array <- as_nanoarrow_array(numbers, schema = schema)
    actual_schema <- infer_nanoarrow_schema(array)
    expect_true(nanoarrow_schema_identical(actual_schema, schema))
    expect_identical(convert_array(array), as.double(numbers))
  }
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
  # The last element here is 0 because (int)nan is undefined behaviour
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(1:10, 0L)))
  )

  # With overflow
  expect_warning(
    as_nanoarrow_array(.Machine$integer.max + as.double(1:5), schema = na_int32()),
    class = "nanoarrow_warning_lossy_conversion"
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

test_that("as_nanoarrow_array() works for double() -> na_float()", {
  # Without nulls
  array <- as_nanoarrow_array(as.double(1:10), schema = na_float())
  expect_identical(infer_nanoarrow_schema(array)$format, "f")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(convert_array(array), as.double(1:10))

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA_real_), schema = na_float())
  expect_identical(infer_nanoarrow_schema(array)$format, "f")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(convert_array(array), c(1:10, NA_real_))
})

test_that("as_nanoarrow_array() works for double() -> na_half_float()", {
  # Without nulls
  array <- as_nanoarrow_array(as.double(1:10), schema = na_half_float())
  expect_identical(infer_nanoarrow_schema(array)$format, "e")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(convert_array(array), as.double(1:10))

  # With nulls
  array <- as_nanoarrow_array(c(1:10, NA_real_), schema = na_half_float())
  expect_identical(infer_nanoarrow_schema(array)$format, "e")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(convert_array(array), c(1:10, NA_real_))
})

test_that("as_nanoarrow_array() works for integer64() -> na_int32()", {
  skip_if_not_installed("bit64")

  # Without nulls
  array <- as_nanoarrow_array(bit64::as.integer64(1:10), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(1:10))
  )

  # With nulls
  array <- as_nanoarrow_array(bit64::as.integer64(c(1:10, NA_real_)), schema = na_int32())
  expect_identical(infer_nanoarrow_schema(array)$format, "i")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  # The last element here is 0 because (int)nan is undefined behaviour
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(1:10, 0L)))
  )
})

test_that("as_nanoarrow_array() works for integer64() -> na_int64()", {
  skip_if_not_installed("bit64")

  # Default roundtrip
  array <- as_nanoarrow_array(bit64::as.integer64(1:10))
  expect_identical(convert_array(array, double()), as.double(1:10))

  # Without nulls
  array <- as_nanoarrow_array(bit64::as.integer64(1:10), schema = na_int64())
  expect_identical(infer_nanoarrow_schema(array)$format, "l")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  # This *is* how we create int64 buffers, so just check the roundtrip
  expect_identical(convert_array(array, double()), as.double(1:10))

  # With nulls
  array <- as_nanoarrow_array(bit64::as.integer64(c(1:10, NA_real_)), schema = na_int64())
  expect_identical(infer_nanoarrow_schema(array)$format, "l")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 10), FALSE, rep(FALSE, 5)))
  )
  expect_identical(convert_array(array, double()), as.double(c(1:10, NA_real_)))
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

test_that("as_nanoarrow_array() works for character() -> na_string_view()", {
  # Without nulls
  array <- as_nanoarrow_array(letters, schema = na_string_view())
  expect_identical(infer_nanoarrow_schema(array)$format, "vu")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  # All these strings are shorter than four characters and thus are all inlined
  expect_identical(length(array$buffers), 3L)
  expect_identical(as.vector(array$buffers[[3]]), double())

  # With nulls
  array <- as_nanoarrow_array(c(letters, NA), schema = na_string_view())
  expect_identical(infer_nanoarrow_schema(array)$format, "vu")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 26), FALSE, rep(FALSE, 5)))
  )
  # All these strings are shorter than four characters and thus are all inlined
  expect_identical(length(array$buffers), 3L)
  expect_identical(as.vector(array$buffers[[3]]), double())

  # With non-inlinable strings
  item <- "this string is longer than 12 bytes"
  array <- as_nanoarrow_array(item, schema = na_string_view())
  expect_identical(length(array$buffers), 4L)
  expect_identical(as.raw(array$buffers[[3]]), charToRaw(item))
  expect_identical(as.vector(array$buffers[[4]]), as.double(nchar(item)))
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

test_that("as_nanoarrow_array() works for Date -> na_date32()", {
  array <- as_nanoarrow_array(as.Date(c("2000-01-01", "2023-02-03", NA)))

  expect_identical(infer_nanoarrow_schema(array)$format, "tdD")
  expect_identical(array$length, 3L)
  expect_identical(array$null_count, 1L)

  expect_identical(as.raw(array$buffers[[1]]), as.raw(0x03))
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(10957L, 19391L, NA)))
  )

  # Sub-day precision handling
  expect_identical(
    as.vector(as_nanoarrow_array(as.Date(c(-0.5, 0, 0.5), origin = as.Date("1970-01-01")))),
    as.Date(c(-1, 0, 0), origin = as.Date("1970-01-01"))
  )
  # Integer based Date support
  expect_identical(
    as.vector(as_nanoarrow_array(as.Date(c(-1L, 0L, 1L), origin = as.Date("1970-01-01")))),
    as.Date(c(-1, 0, 1), origin = as.Date("1970-01-01"))
  )
})

test_that("as_nanoarrow_array() works for Date -> na_date64()", {
  array <- as_nanoarrow_array(
    as.Date(c("2000-01-01", "2023-02-03", NA)),
    schema = na_date64()
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "tdm")
  expect_identical(array$length, 3L)
  expect_identical(array$null_count, 1L)

  expect_identical(as.raw(array$buffers[[1]]), as.raw(0x03))
  storage <- as_nanoarrow_array(
    c(10957L, 19391L, NA) * 86400000,
    schema = na_int64()
  )

  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(storage$buffers[[2]])
  )
})

test_that("as_nanoarrow_array() works for POSIXct -> na_timestamp()", {
  array <- as_nanoarrow_array(
    as.POSIXct(c("2000-01-01", "2023-02-03", NA), tz = "UTC"),
    schema = na_timestamp("ms", timezone = "UTC")
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "tsm:UTC")
  expect_identical(array$length, 3L)
  expect_identical(array$null_count, 1L)

  expect_identical(as.raw(array$buffers[[1]]), as.raw(0x03))
  storage <- as_nanoarrow_array(
    c(10957L, 19391L, NA) * 86400000,
    schema = na_int64()
  )

  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(storage$buffers[[2]])
  )
})

test_that("as_nanoarrow_array() works for difftime -> na_duration()", {
  array <- as_nanoarrow_array(
    as.difftime(c(1:5, NA), units = "secs"),
    schema = na_duration("ms")
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "tDm")
  expect_identical(array$length, 6L)
  expect_identical(array$null_count, 1L)

  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 5), FALSE, rep(FALSE, 2)))
  )
  storage <- as_nanoarrow_array(
    c(1:5, NA) * 1000,
    schema = na_int64()
  )

  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(storage$buffers[[2]])
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

test_that("as_nanoarrow_array() works for matrix -> na_fixed_size_list()", {
  mat <- matrix(1:6, ncol = 2, byrow = TRUE)

  # Check without explicit schema
  array <- as_nanoarrow_array(mat)
  expect_identical(infer_nanoarrow_schema(array)$format, "+w:2")
  expect_identical(infer_nanoarrow_schema(array)$children[[1]]$format, "i")
  expect_identical(array$buffers[[1]]$size_bytes, 0)
  expect_identical(convert_buffer(array$children[[1]]$buffers[[2]]), 1:6)

  # Check with explicit schema
  array <- as_nanoarrow_array(
    mat,
    schema = na_fixed_size_list(na_double(), list_size = 2)
  )
  expect_identical(infer_nanoarrow_schema(array)$format, "+w:2")
  expect_identical(infer_nanoarrow_schema(array)$children[[1]]$format, "g")
  expect_identical(array$buffers[[1]]$size_bytes, 0)
  expect_identical(convert_buffer(array$children[[1]]$buffers[[2]]), as.double(1:6))
})

test_that("as_nanoarrow_array() works for blob::blob() -> na_fixed_size_binary()", {
  # Without nulls
  array <- as_nanoarrow_array(blob::as_blob(letters), schema = na_fixed_size_binary(1))
  expect_identical(infer_nanoarrow_schema(array)$format, "w:1")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(paste(letters, collapse = "")))
  )

  # With nulls
  array <- as_nanoarrow_array(
    blob::as_blob(c(letters, NA)),
    schema = na_fixed_size_binary(1)
  )
  expect_identical(infer_nanoarrow_schema(array)$format, "w:1")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 26), FALSE, rep(FALSE, 5)))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    c(as.raw(as_nanoarrow_buffer(paste(letters, collapse = ""))), as.raw(0x00))
  )
})

test_that("as_nanoarrow_array() works for blob::blob() -> na_large_binary()", {
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

test_that("as_nanoarrow_array() works for blob::blob() -> na_binary_view()", {
  # Without nulls
  array <- as_nanoarrow_array(blob::as_blob(letters), schema = na_binary_view())
  expect_identical(infer_nanoarrow_schema(array)$format, "vz")
  expect_identical(as.raw(array$buffers[[1]]), raw())
  expect_identical(array$offset, 0L)
  expect_identical(array$null_count, 0L)
  # All these strings are shorter than four characters and thus are all inlined
  expect_identical(length(array$buffers), 3L)
  expect_identical(as.vector(array$buffers[[3]]), double())

  # With nulls
  array <- as_nanoarrow_array(
    blob::as_blob(c(letters, NA)),
    schema = na_binary_view()
  )
  expect_identical(infer_nanoarrow_schema(array)$format, "vz")
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    packBits(c(rep(TRUE, 26), FALSE, rep(FALSE, 5)))
  )
  # All these strings are shorter than four characters and thus are all inlined
  expect_identical(length(array$buffers), 3L)
  expect_identical(as.vector(array$buffers[[3]]), double())

  # With non-inlinable strings
  item <- list(charToRaw("this string is longer than 12 bytes"))
  array <- as_nanoarrow_array(item, schema = na_binary_view())
  expect_identical(length(array$buffers), 4L)
  expect_identical(as.raw(array$buffers[[3]]), item[[1]])
  expect_identical(as.vector(array$buffers[[4]]), as.double(length(item[[1]])))
})

test_that("as_nanoarrow_array() works for list(raw()) -> na_binary()", {
  # Without nulls
  array <- as_nanoarrow_array(lapply(letters, charToRaw))
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
  array <- as_nanoarrow_array(c(lapply(letters, charToRaw), list(NULL)))
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

test_that("as_nanoarrow_array() works for list(NULL) -> na_list(na_na())", {
  array <- as_nanoarrow_array(list(NULL))
  expect_identical(infer_nanoarrow_schema(array)$format, "+l")
  expect_identical(array$length, 1L)
  expect_identical(array$null_count, 1L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    as.raw(as_nanoarrow_array(FALSE)$buffers[[2]])
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(0L, 0L)))
  )
  expect_identical(infer_nanoarrow_schema(array$children[[1]])$format, "n")
  expect_identical(array$children[[1]]$length, 0L)
})

test_that("as_nanoarrow_array() works for list(integer()) -> na_list(na_int32())", {
  array <- as_nanoarrow_array(list(1:5, 6:10), schema = na_list(na_int32()))
  expect_identical(infer_nanoarrow_schema(array)$format, "+l")
  expect_identical(array$length, 2L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    as.raw(as_nanoarrow_array(c(TRUE, TRUE))$buffers[[2]])
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(0L, 5L, 10L)))
  )
  expect_identical(infer_nanoarrow_schema(array$children[[1]])$format, "i")
  expect_identical(array$children[[1]]$length, 10L)
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

test_that("as_nanoarrow_array() can convert data.frame() to sparse_union()", {
  # Features: At least one element with more than one non-NA value,
  # one element with all NA values.
  test_df <- data.frame(
    lgl = c(TRUE, NA, NA, NA, NA, FALSE),
    int = c(NA, 123L, NA, NA, NA, NA),
    dbl = c(NA, NA, 456, NA, NA, NA),
    chr = c(NA, NA, NA, "789", NA, NA),
    stringsAsFactors = FALSE
  )

  array <- as_nanoarrow_array(
    test_df,
    schema = na_sparse_union(lapply(test_df, infer_nanoarrow_schema))
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "+us:0,1,2,3")
  expect_identical(array$length, 6L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    as.raw(as_nanoarrow_buffer(as.raw(c(0L, 1L, 2L, 3L, 0L, 0L))))
  )

  expect_identical(
    lapply(array$children, convert_array),
    lapply(test_df, identity)
  )
  expect_identical(convert_array(array), test_df)
})

test_that("as_nanoarrow_array() can convert data.frame() to sparse_union()", {
  test_df <- data.frame(
    lgl = c(TRUE, NA, NA, NA, NA, FALSE),
    int = c(NA, 123L, NA, NA, NA, NA),
    dbl = c(NA, NA, 456, NA, NA, NA),
    chr = c(NA, NA, NA, "789", NA, NA),
    stringsAsFactors = FALSE
  )

  array <- as_nanoarrow_array(
    test_df,
    schema = na_dense_union(lapply(test_df, infer_nanoarrow_schema))
  )

  expect_identical(infer_nanoarrow_schema(array)$format, "+ud:0,1,2,3")
  expect_identical(array$length, 6L)
  expect_identical(array$null_count, 0L)
  expect_identical(
    as.raw(array$buffers[[1]]),
    as.raw(as_nanoarrow_buffer(as.raw(c(0L, 1L, 2L, 3L, 0L, 0L))))
  )
  expect_identical(
    as.raw(array$buffers[[2]]),
    as.raw(as_nanoarrow_buffer(c(0L, 0L, 0L, 0L, 1L, 2L)))
  )

  expect_identical(
    lapply(array$children, convert_array),
    list(
      lgl = c(TRUE, NA, FALSE),
      int = 123L,
      dbl = 456,
      chr = "789"
    )
  )
  expect_identical(convert_array(array), test_df)
})

test_that("as_nanoarrow_array() for union type errors for unsupported objects", {
  expect_error(
    as_nanoarrow_array(data.frame(), schema = na_dense_union()),
    "Can't convert data frame with 0 columns"
  )
})

test_that("storage_integer_for_decimal generates the correct string output", {
  numbers <- c(
    0, 1.23, 4, -1/3, .Machine$double.eps,
    123.4567890, -123.4567890, NA
  )

  expect_identical(
    storage_decimal_for_decimal(numbers, 4),
    c(
      "0.0000", "1.2300", "4.0000", "-0.3333", "0.0000",
      "123.4568", "-123.4568", NA
    )
  )

  expect_identical(
    storage_decimal_for_decimal(numbers, 0),
    c(
      "0", "1", "4", "0", "0",
      "123", "-123", NA
    )
  )

  expect_identical(
    storage_decimal_for_decimal(numbers, -1),
    c(
      "0", "0", "0", "0", "0",
      "12", "-12", NA
    )
  )

  expect_identical(
    storage_integer_for_decimal(numbers, 4),
    c(
      "00000", "12300", "40000", "-03333", "00000",
      "1234568", "-1234568", NA
    )
  )

  # Check that we're generating the output we think we're generating
  # with a random sample of numbers at full precision and a random sample
  # of numbers at low precision.
  withr::with_seed(4958, {
    numbers <- runif(1000, -1000, 1000)
    for (scale in 1:15) {
      expect_match(
        storage_decimal_for_decimal(numbers, scale),
        paste0("-?[0-9]+\\.[0-9]{", scale, "}")
      )
    }

    # Also check that we have the required number of digits after the decimal
    # when the numbers start out not having no digits after the decimal
    numbers <- round(numbers)
    for (scale in 1:15) {
      expect_match(
        storage_decimal_for_decimal(numbers, scale),
        paste0("-?[0-9]+\\.[0-9]{", scale, "}")
      )
    }
  })
})
