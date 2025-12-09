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

test_that("convert_array() errors for invalid arrays", {
  array <- as_nanoarrow_array(1:10)
  nanoarrow_array_set_schema(
    array,
    na_string(),
    validate = FALSE
  )

  expect_error(
    convert_array(array),
    "Expected array with 3 buffer"
  )
})

test_that("convert_array() errors for unsupported ptype", {
  array <- as_nanoarrow_array(1:10)

  # an S3 unsupported type
  expect_error(
    convert_array(array, structure(list(), class = "some_class")),
    "Can't convert array <int32> to R vector of type some_class"
  )

  # A non-S3 unsupported type
  expect_error(
    convert_array(array, environment()),
    "Can't convert array <int32> to R vector of type environment"
  )

  # An array with a name to an unsupported type
  struct_array <- as_nanoarrow_array(data.frame(x = 1L))
  expect_error(
    convert_array(struct_array$children$x, environment()),
    "Can't convert `x`"
  )
})

test_that("convert_array() errors for unsupported array", {
  unsupported_array <- nanoarrow_array_init(na_interval_day_time())
  expect_error(
    convert_array(as_nanoarrow_array(unsupported_array)),
    "Can't infer R vector type for <interval_day_time>"
  )
})

test_that("convert to vector works for data.frame", {
  df <- data.frame(a = 1L, b = "two", c = 3, d = TRUE, stringsAsFactors = FALSE)
  array <- as_nanoarrow_array(df)

  expect_identical(convert_array(array, NULL), df)
  expect_identical(convert_array(array, df), df)

  expect_error(
    convert_array(array, data.frame(a = integer(), b = raw())),
    "Expected data.frame\\(\\) ptype with 4 column\\(s\\) but found 2 column\\(s\\)"
  )

  bad_ptype <- data.frame(a = integer(), b = raw(), c = double(), d = integer())
  expect_error(
    convert_array(array, bad_ptype),
    "Can't convert `b` <string> to R vector of type raw"
  )
})

test_that("convert to vector works for extension<struct> -> data.frame()", {
  array <- nanoarrow_extension_array(
    data.frame(x = c(TRUE, FALSE, NA, FALSE, TRUE)),
    "some_ext"
  )

  expect_warning(
    expect_identical(
      convert_array(array, data.frame(x = logical())),
      data.frame(x = c(TRUE, FALSE, NA, FALSE, TRUE))
    ),
    "Converting unknown extension"
  )
})

test_that("convert to vector works for dictionary<struct> -> data.frame()", {
  array <- as_nanoarrow_array(c(0L, 1L, 2L, 1L, 0L))
  array$dictionary <- as_nanoarrow_array(data.frame(x = c(TRUE, FALSE, NA)))

  expect_identical(
    convert_array(array, data.frame(x = logical())),
    data.frame(x = c(TRUE, FALSE, NA, FALSE, TRUE))
  )
})

test_that("convert to vector works for function()", {
  skip_if_not_installed("tibble")

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
  array_nested <- as_nanoarrow_array(df_nested_df)
  expect_identical(
    convert_array(array_nested, tibble_or_bust),
    tibble::tibble(a = 1L, b = "two", c = tibble::tibble(a = 3))
  )
})

test_that("convert to vector works for tibble", {
  skip_if_not_installed("tibble")

  array <- as_nanoarrow_array(
    data.frame(a = 1L, b = "two", stringsAsFactors = FALSE)
  )
  expect_identical(
    convert_array(array, tibble::tibble(a = integer(), b = character())),
    tibble::tibble(a = 1L, b = "two")
  )

  # Check nested tibble at both levels
  tbl_nested_df <- tibble::tibble(a = 1L, b = "two", c = data.frame(a = 3))
  array_nested <- as_nanoarrow_array(tbl_nested_df)

  expect_identical(
    convert_array(array_nested, tbl_nested_df),
    tbl_nested_df
  )

  df_nested_tbl <- as.data.frame(tbl_nested_df)
  df_nested_tbl$c <- tibble::as_tibble(df_nested_tbl$c)
  expect_identical(
    convert_array(array_nested, df_nested_tbl),
    df_nested_tbl
  )
})

test_that("convert to vector works for a rcrd-sytle vctr with complex columns", {
  skip_if_not_installed("vctrs")

  rcrd <- vctrs::new_rcrd(list(x = data.frame(y = 1:10)))
  rcrd_array <- as_nanoarrow_array(vctrs::vec_data(rcrd))
  expect_identical(
    convert_array(rcrd_array, rcrd),
    rcrd
  )
})

test_that("convert to vector works for nanoarrow_vctr()", {
  array <- as_nanoarrow_array(c("one", "two", "three"))

  # Check implicit/inferred nanoarrow_vctr() schema
  vctr <- convert_array(array, nanoarrow_vctr())
  expect_s3_class(vctr, "nanoarrow_vctr")
  expect_length(vctr, 3)
  schema <- infer_nanoarrow_schema(vctr)
  expect_identical(schema$format, "u")

  # Check with explicit schema of the correct type
  vctr <- convert_array(array, nanoarrow_vctr(na_string()))
  expect_s3_class(vctr, "nanoarrow_vctr")
  expect_length(vctr, 3)
  schema <- infer_nanoarrow_schema(vctr)
  expect_identical(schema$format, "u")

  # Check conversion of a struct array
  df <- data.frame(x = c("one", "two", "three"))
  array <- as_nanoarrow_array(df)

  vctr <- convert_array(array, nanoarrow_vctr())
  expect_s3_class(vctr, "nanoarrow_vctr")
  expect_length(vctr, 3)
  schema <- infer_nanoarrow_schema(vctr)
  expect_identical(schema$format, "+s")

  vctr <- convert_array(array, nanoarrow_vctr(na_struct(list(x = na_string()))))
  expect_s3_class(vctr, "nanoarrow_vctr")
  expect_length(vctr, 3)
  schema <- infer_nanoarrow_schema(vctr)
  expect_identical(schema$format, "+s")
})

test_that("batched convert to vector works for nanoarrow_vctr()", {
  empty_stream <- basic_array_stream(list(), schema = na_string())
  empty_vctr <- convert_array_stream(empty_stream, nanoarrow_vctr())
  expect_length(empty_vctr, 0)
  expect_identical(infer_nanoarrow_schema(empty_vctr)$format, "u")

  stream1 <- basic_array_stream(list(c("one", "two", "three")))
  vctr1 <- convert_array_stream(stream1, nanoarrow_vctr())
  expect_length(vctr1, 3)

  stream2 <- basic_array_stream(
    list(c("one", "two", "three"), c("four", "five", "six", "seven"))
  )
  vctr2 <- convert_array_stream(stream2, nanoarrow_vctr())
  expect_length(vctr2, 7)
  expect_identical(
    convert_array_stream(as_nanoarrow_array_stream(vctr2)),
    c("one", "two", "three", "four", "five", "six", "seven")
  )
})

test_that("convert to vector works for data.frame(nanoarrow_vctr())", {
  array <- as_nanoarrow_array(data.frame(x = 1:5))
  df_vctr <- convert_array(array, data.frame(x = nanoarrow_vctr()))
  expect_s3_class(df_vctr$x, "nanoarrow_vctr")
  expect_identical(
    convert_array_stream(as_nanoarrow_array_stream(df_vctr$x)),
    1:5
  )
})

test_that("convert to vector works for list_of(nanoarrow_vctr())", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  array <- as_nanoarrow_array(
    list(1:5, 6:10, NULL, 11:13),
    schema = na_list(na_int32())
  )

  list_vctr <- convert_array(array, vctrs::list_of(nanoarrow_vctr()))

  # Each item in the list should be a vctr with one chunk that is a slice
  # of the original array
  expect_s3_class(list_vctr[[1]], "nanoarrow_vctr")
  vctr_array <- attr(list_vctr[[1]], "chunks")[[1]]
  expect_identical(vctr_array$offset, 0L)
  expect_identical(vctr_array$length, 5L)
  expect_identical(convert_buffer(vctr_array$buffers[[2]]), 1:5)

  expect_s3_class(list_vctr[[2]], "nanoarrow_vctr")
  vctr_array <- attr(list_vctr[[2]], "chunks")[[1]]
  expect_identical(vctr_array$offset, 5L)
  expect_identical(vctr_array$length, 5L)
  expect_identical(convert_buffer(vctr_array$buffers[[2]]), 1:10)

  expect_null(list_vctr[[3]])

  expect_s3_class(list_vctr[[4]], "nanoarrow_vctr")
  vctr_array <- attr(list_vctr[[4]], "chunks")[[1]]
  expect_identical(vctr_array$offset, 10L)
  expect_identical(vctr_array$length, 3L)
  expect_identical(convert_buffer(vctr_array$buffers[[2]]), 1:13)
})

test_that("batched convert to vector works for nanoarrow_vctr() keeps subclass", {
  vctr_ptype <- nanoarrow_vctr(subclass = "some_subclass")

  empty_stream <- basic_array_stream(list(), schema = na_string())
  empty_vctr <- convert_array_stream(empty_stream, vctr_ptype)
  expect_s3_class(empty_vctr, "some_subclass")

  stream1 <- basic_array_stream(list(c("")))
  vctr1 <- convert_array_stream(stream1, vctr_ptype)
  expect_s3_class(vctr1, "some_subclass")

  stream2 <- basic_array_stream(list(c(""), c("")))
  vctr2 <- convert_array_stream(stream2, vctr_ptype)
  expect_s3_class(vctr2, "some_subclass")
})

test_that("convert to vector works for struct-style vectors", {
  array <- as_nanoarrow_array(as.POSIXlt("2021-01-01", tz = "America/Halifax"))
  expect_identical(
    convert_array(array),
    as.data.frame(
      unclass(as.POSIXlt("2021-01-01", tz = "America/Halifax")),
      stringsAsFactors = FALSE
    )
  )

  array <- as_nanoarrow_array(as.POSIXlt("2021-01-01", tz = "America/Halifax"))
  expect_identical(
    convert_array(array, as.POSIXlt("2021-01-01", tz = "America/Halifax")),
    as.POSIXlt("2021-01-01", tz = "America/Halifax")
  )
})

test_that("convert to vector works for unspecified()", {
  skip_if_not_installed("vctrs")

  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  # implicit for null type
  expect_identical(
    convert_array(array, to = NULL),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for null type
  expect_identical(
    convert_array(array, vctrs::unspecified()),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for non-null type that is all NAs
  array <- as_nanoarrow_array(rep(NA_integer_, 10))
  expect_identical(
    convert_array(array, vctrs::unspecified()),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for non-null type that is not all NAs
  array <- as_nanoarrow_array(c(1L, rep(NA_integer_, 9)))
  expect_warning(
    expect_identical(
      convert_array(array, vctrs::unspecified()),
      vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
    ),
    class = "nanoarrow_warning_lossy_conversion"
  )
})

test_that("convert to vector works for valid logical()", {
  skip_if_not_installed("arrow")

  arrow_numeric_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64(),
    uint64 = arrow::uint64(),
    float32 = arrow::float32(),
    float64 = arrow::float64()
  )

  vals <- c(NA, 0:10)
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals, schema = arrow_numeric_types[[!!nm]]),
        logical()
      ),
      vals != 0
    )
  }

  vals_no_na <- 0:10
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals_no_na, schema = arrow_numeric_types[[!!nm]]),
        logical()
      ),
      vals_no_na != 0
    )
  }

  # Boolean array to logical
  expect_identical(
    convert_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      logical()
    ),
    c(NA, TRUE, FALSE)
  )

  expect_identical(
    convert_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      logical()
    ),
    c(TRUE, FALSE)
  )
})

test_that("convert to vector works for null -> logical()", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, logical()),
    rep(NA, 10)
  )
})

test_that("convert to vector works for extension<boolean> -> logical()", {
  array <- nanoarrow_extension_array(c(TRUE, FALSE, NA), "some_ext")

  expect_warning(
    expect_identical(
      convert_array(array, logical()),
      c(TRUE, FALSE, NA)
    ),
  "Converting unknown extension"
  )
})

test_that("convert to vector works for dictionary<boolean> -> logical()", {
  array <- as_nanoarrow_array(c(0L, 1L, 2L, 1L, 0L))
  array$dictionary <- as_nanoarrow_array(c(TRUE, FALSE, NA))

  expect_identical(
    convert_array(array, logical()),
    c(TRUE, FALSE, NA, FALSE, TRUE)
  )
})

test_that("convert to vector errors for bad array to logical()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), logical()),
    "Can't convert array <string> to R vector of type logical"
  )
})

test_that("convert to vector works for valid integer()", {
  skip_if_not_installed("arrow")

  arrow_int_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64(),
    uint64 = arrow::uint64(),
    float32 = arrow::float32(),
    float64 = arrow::float64()
  )

  ints <- c(NA, 0:10)
  for (nm in names(arrow_int_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(ints, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints
    )
  }

  ints_no_na <- 0:10
  for (nm in names(arrow_int_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(ints_no_na, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints_no_na
    )
  }

  # Boolean array to integer
  expect_identical(
    convert_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(NA, 1L, 0L)
  )

  expect_identical(
    convert_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(1L, 0L)
  )
})

test_that("convert to works for integer() -> character()", {
  skip_if_not_installed("arrow")

  arrow_int_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64()
  )

  ints <- c(NA, 0:10)
  for (nm in names(arrow_int_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(ints, schema = arrow_int_types[[!!nm]]),
        character()
      ),
      as.character(ints)
    )
  }
})

test_that("convert to vector works for null -> logical()", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, integer()),
    rep(NA_integer_, 10)
  )
})

test_that("convert to vector works for extension<integer> -> integer()", {
  array <- nanoarrow_extension_array(c(0L, 1L, NA_integer_), "some_ext")

  expect_warning(
    expect_identical(
      convert_array(array, integer()),
      c(0L, 1L, NA_integer_)
    ),
    "Converting unknown extension"
  )
})

test_that("convert to vector warns for invalid integer()", {
  array <- as_nanoarrow_array(.Machine$integer.max + 1)
  expect_warning(
    expect_identical(convert_array(array, integer()), NA_integer_),
    class = "nanoarrow_warning_lossy_conversion"
  )

  array <- as_nanoarrow_array(c(NA, .Machine$integer.max + 1))
  expect_warning(
    expect_identical(convert_array(array, integer()), c(NA_integer_, NA_integer_)),
    class = "nanoarrow_warning_lossy_conversion"
  )
})

test_that("convert to vector errors for bad array to integer()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), integer()),
    "Can't convert array <string> to R vector of type integer"
  )
})

test_that("convert to vector works for valid double()", {
  skip_if_not_installed("arrow")

  arrow_numeric_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64(),
    uint64 = arrow::uint64(),
    float16 = arrow::float16(),
    float32 = arrow::float32(),
    float64 = arrow::float64()
  )

  vals <- as.double(c(NA, 0:10))
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals, schema = arrow_numeric_types[[!!nm]]),
        double()
      ),
      vals
    )
  }

  vals_no_na <- as.double(0:10)
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals_no_na, schema = arrow_numeric_types[[!!nm]]),
        double()
      ),
      vals_no_na
    )
  }

  # Boolean array to double
  expect_identical(
    convert_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      double()
    ),
    as.double(c(NA, 1L, 0L))
  )

  expect_identical(
    convert_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      double()
    ),
    as.double(c(1L, 0L))
  )
})

test_that("convert to vector works for decimal -> double()", {
  constructors <- list(na_decimal32, na_decimal64, na_decimal256, na_decimal128)
  for (constructor in constructors) {
    numbers <- round(c(pi, -pi, 123.4567, NA, -123.4567, 0, 123), 4)

    # Check scale of 4 (min required for lossless roundtrip)
    array4 <- as_nanoarrow_array(numbers, schema = constructor(9, 4))
    expect_identical(convert_array(array4), numbers)

    # Check scale of 5 (requires adding some zeroes in C)
    array5 <- as_nanoarrow_array(numbers, schema = constructor(9, 5))
    expect_identical(convert_array(array5), numbers)

    # Check negative scale (also requires adding some zeroes in C)
    numbers_neg_scale <- c(12300, -12300, 0, NA, 100)
    array_neg2 <- as_nanoarrow_array(numbers_neg_scale, schema = constructor(9, -2))
    expect_identical(convert_array(array_neg2), numbers_neg_scale)
  }

  # Make sure we agree with arrow
  skip_if_not_installed("arrow")

  expect_identical(as.vector(arrow::as_arrow_array(array4)), numbers)
  expect_identical(as.vector(arrow::as_arrow_array(array5)), numbers)
  expect_identical(as.vector(arrow::as_arrow_array(array_neg2)), numbers_neg_scale)
})

test_that("convert to vector works for null -> double()", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, double()),
    rep(NA_real_, 10)
  )
})

test_that("convert to vector works for extension<double> -> double()", {
  array <- nanoarrow_extension_array(c(0, 1, NA_real_), "some_ext")

  expect_warning(
    expect_identical(
      convert_array(array, double()),
      c(0, 1, NA_real_)
    ),
    "Converting unknown extension"
  )
})

test_that("convert to vector works for dictionary<double> -> double()", {
  array <- as_nanoarrow_array(c(0L, 1L, 2L, 1L, 0L))
  array$dictionary <- as_nanoarrow_array(c(123, 0,  NA_real_))

  expect_identical(
    convert_array(array, double()),
    c(123, 0, NA_real_, 0, 123)
  )
})

test_that("convert to vector warns for possibly invalid double()", {
  array <- as_nanoarrow_array(2^54, schema = na_int64())
  expect_warning(
    convert_array(array, double()),
    class = "nanoarrow_warning_lossy_conversion"
  )
})

test_that("convert to vector errors for bad array to double()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), double()),
    "Can't convert array <string> to R vector of type numeric"
  )
})

test_that("convert to vector works for valid integer64()", {
  skip_if_not_installed("bit64")
  skip_if_not_installed("arrow")

  arrow_numeric_types <- list(
    int8 = arrow::int8(),
    uint8 = arrow::uint8(),
    int16 = arrow::int16(),
    uint16 = arrow::uint16(),
    int32 = arrow::int32(),
    uint32 = arrow::uint32(),
    int64 = arrow::int64(),
    uint64 = arrow::uint64(),
    float32 = arrow::float32(),
    float64 = arrow::float64()
  )

  vals <- bit64::as.integer64(c(NA, 0:10))
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals, schema = arrow_numeric_types[[!!nm]]),
        bit64::integer64()
      ),
      vals
    )
  }

  vals_no_na <- bit64::as.integer64(0:10)
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      convert_array(
        as_nanoarrow_array(vals_no_na, schema = arrow_numeric_types[[!!nm]]),
        bit64::integer64()
      ),
      vals_no_na
    )
  }

  # Boolean array to double
  expect_identical(
    convert_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      bit64::integer64()
    ),
    bit64::as.integer64(c(NA, 1L, 0L))
  )

  expect_identical(
    convert_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      bit64::integer64()
    ),
    bit64::as.integer64(c(1L, 0L))
  )
})

test_that("convert to vector works for null -> integer64()", {
  skip_if_not_installed("bit64")

  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, bit64::integer64()),
    rep(bit64::NA_integer64_, 10)
  )
})

test_that("convert to vector works for extension<int64> -> integer64()", {
  skip_if_not_installed("bit64")

  vec <- bit64::as.integer64(c(0, 1, NA))
  array <- nanoarrow_extension_array(vec, "some_ext")

  expect_warning(
    expect_identical(
      convert_array(array, bit64::integer64()),
      vec
    ),
    "Converting unknown extension"
  )
})

test_that("convert to vector errors for bad array to integer64()", {
  skip_if_not_installed("bit64")

  expect_error(
    convert_array(as_nanoarrow_array(letters), bit64::integer64()),
    "Can't convert array <string> to R vector of type integer64"
  )
})

test_that("convert to vector works for character()", {
  array <- as_nanoarrow_array(letters)
  expect_identical(
    convert_array(array, character()),
    letters
  )

  # make sure we get altrep here
  expect_true(is_nanoarrow_altrep(convert_array(array, character())))

  # check an array that we can't convert
  expect_error(
    convert_array(as_nanoarrow_array(1:5), list()),
    "Can't convert array <int32> to R vector of type list"
  )
})

test_that("convert to vector works for string_view -> character()", {
  array <- as_nanoarrow_array(letters, schema = na_string_view())
  expect_identical(
    convert_array(array, character()),
    letters
  )

  array_with_nulls <- as_nanoarrow_array(c(letters, NA), schema = na_string_view())
  expect_identical(
    convert_array(array_with_nulls, character()),
    c(letters, NA)
  )
})

test_that("convert to vector works for null -> character()", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  all_nulls <- convert_array(array, character())
  nanoarrow_altrep_force_materialize(all_nulls)
  expect_identical(
    all_nulls,
    rep(NA_character_, 10)
  )
})

test_that("convert to vector works for extension<string> -> character()", {
  array <- nanoarrow_extension_array(c("a", "b", NA_character_), "some_ext")

  expect_warning(
    expect_identical(
      convert_array(array, character()),
      c("a", "b", NA_character_)
    ),
    "Converting unknown extension"
  )
})

test_that("convert to vector works for dictionary<string> -> character()", {
  array <- as_nanoarrow_array(factor(letters[5:1]))

  # Via S3 dispatch
  expect_identical(
    convert_array(array, character()),
    c("e", "d", "c", "b", "a")
  )

  # Via C -> S3 dispatch
  expect_identical(
    convert_array.default(array, character()),
    c("e", "d", "c", "b", "a")
  )
})

test_that("convert to vector works for dictionary<string> -> factor()", {
  array <- as_nanoarrow_array(factor(letters[5:1]))

  # With empty levels
  expect_identical(
    convert_array(array, factor()),
    factor(letters[5:1])
  )

  # With identical levels
  expect_identical(
    convert_array(array, factor(levels = c("a", "b", "c", "d", "e"))),
    factor(letters[5:1])
  )

  # With mismatched levels
  expect_identical(
    convert_array(array, factor(levels = c("b", "a", "c", "e", "d"))),
    factor(letters[5:1], levels = c("b", "a", "c", "e", "d"))
  )

  expect_error(
    convert_array(array, factor(levels = letters[-4])),
    "some levels in data do not exist"
  )
})

test_that("convert to vector works for decimal -> character()", {
  constructors <- list(na_decimal32, na_decimal64, na_decimal256, na_decimal128)
  for (constructor in constructors) {
    numbers <- round(c(pi, -pi, 123.4567, NA, -123.4567, 0, 123), 4)

    # Check scale of 4 (min required for lossless roundtrip)
    array4 <- as_nanoarrow_array(numbers, schema = constructor(9, 4))
    expect_identical(
      convert_array(array4, character()),
      c("3.1416", "-3.1416", "123.4567", NA_character_,
        "-123.4567", "0.0000", "123.0000")
    )

    # Check scale of 5 (requires adding some zeroes in C)
    array5 <- as_nanoarrow_array(numbers, schema = constructor(9, 5))
    expect_identical(
      convert_array(array5, character()),
      c("3.14160", "-3.14160", "123.45670", NA_character_,
        "-123.45670", "0.00000", "123.00000")
    )

    # Check negative scale (also requires adding some zeroes in C)
    numbers_neg_scale <- c(12300, -12300, 0, NA, 100)
    array_neg2 <- as_nanoarrow_array(numbers_neg_scale, schema = constructor(9, -2))
    expect_identical(
      convert_array(array_neg2, character()),
      c("12300", "-12300", "000", NA_character_, "100")
    )
  }
})

test_that("batched convert to vector works for dictionary<string> -> factor()", {
  # A slightly different path: convert_array.factor() called from C multiple
  # times with different dictionaries each time.
  array1 <- as_nanoarrow_array(factor(letters[1:5]))
  array2 <- as_nanoarrow_array(factor(letters[6:10]))
  array3 <- as_nanoarrow_array(factor(letters[11:15]))

  stream <- basic_array_stream(list(array1, array2, array3))
  expect_identical(
    convert_array_stream(stream, factor(levels = letters)),
    factor(letters[1:15], levels = letters)
  )
})

test_that("batched convert to vector errors for dictionary<string> -> factor()", {
  # We can't currently handle a preallocate + fill style conversion where the
  # result is partial_factor().
  array1 <- as_nanoarrow_array(factor(letters[1:5]))
  array2 <- as_nanoarrow_array(factor(letters[6:10]))
  array3 <- as_nanoarrow_array(factor(letters[11:15]))

  stream <- basic_array_stream(list(array1, array2, array3))
  expect_error(
    convert_array_stream(stream, factor()),
    "Can't allocate ptype of class 'factor'"
  )
})

test_that("convert to vector works for blob::blob()", {
  skip_if_not_installed("blob")

  array <- as_nanoarrow_array(list(as.raw(1:5)), schema = na_binary())

  expect_identical(
    convert_array(array),
    blob::blob(as.raw(1:5))
  )

  expect_identical(
    convert_array(array, blob::blob()),
    blob::blob(as.raw(1:5))
  )
})

test_that("convert to vector works for binary_view -> blob::blob()", {
  skip_if_not_installed("blob")

  array <- as_nanoarrow_array(letters, schema = na_binary_view())
  expect_identical(
    convert_array(array, blob::blob()),
    blob::as_blob(lapply(letters, charToRaw))
  )

  array_with_nulls <- as_nanoarrow_array(c(letters, NA), schema = na_binary_view())
  expect_identical(
    convert_array(array_with_nulls, blob::blob()),
    blob::as_blob(c(lapply(letters, charToRaw), list(NULL)))
  )
})

test_that("convert to vector works for null -> blob::blob()", {
  skip_if_not_installed("blob")

  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, blob::blob()),
    blob::new_blob(rep(list(NULL), 10))
  )
})

test_that("convert to vector works for list -> vctrs::list_of", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  array_list <- as_nanoarrow_array(
    arrow::Array$create(
      list(1:5, 6:10, NULL),
      type = arrow::list_of(arrow::int32())
    )
  )

  # Default conversion
  expect_identical(
    convert_array(array_list),
    vctrs::list_of(1:5, 6:10, NULL, .ptype = integer())
  )

  # With explicit ptype
  expect_identical(
    convert_array(array_list, vctrs::list_of(.ptype = double())),
    vctrs::list_of(as.double(1:5), as.double(6:10), NULL, .ptype = double())
  )

  # With bad ptype
  expect_error(
    convert_array(array_list, vctrs::list_of(.ptype = list())),
    "Can't convert `item`"
  )

  # With malformed ptype
  ptype <- vctrs::list_of(.ptype = character())
  attr(ptype, "ptype") <- NULL
  expect_error(
    convert_array(array_list, ptype),
    "Expected attribute 'ptype'"
  )
})

test_that("convert to vector works for large_list -> vctrs::list_of", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  array_list <- as_nanoarrow_array(
    arrow::Array$create(
      list(1:5, 6:10, NULL),
      type = arrow::large_list_of(arrow::int32())
    )
  )

  # Default conversion
  expect_identical(
    convert_array(array_list),
    vctrs::list_of(1:5, 6:10, NULL, .ptype = integer())
  )

  # With explicit ptype
  expect_identical(
    convert_array(array_list, vctrs::list_of(.ptype = double())),
    vctrs::list_of(as.double(1:5), as.double(6:10), NULL, .ptype = double())
  )

  # With bad ptype
  expect_error(
    convert_array(array_list, vctrs::list_of(.ptype = list())),
    "Can't convert `item`"
  )
})

test_that("convert to vector works for fixed_size_list -> vctrs::list_of", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  array_list <- as_nanoarrow_array(
    arrow::Array$create(
      list(1:5, 6:10, NULL),
      type = arrow::fixed_size_list_of(arrow::int32(), 5)
    )
  )

  # Default conversion
  expect_identical(
    convert_array(array_list),
    vctrs::list_of(1:5, 6:10, NULL, .ptype = integer())
  )

  # With explicit ptype
  expect_identical(
    convert_array(array_list, vctrs::list_of(.ptype = double())),
    vctrs::list_of(as.double(1:5), as.double(6:10), NULL, .ptype = double())
  )

  # With bad ptype
  expect_error(
    convert_array(array_list, vctrs::list_of(.ptype = list())),
    "Can't convert `item`"
  )
})

test_that("convert to vector works for null -> vctrs::list_of()", {
  skip_if_not_installed("vctrs")

  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, vctrs::list_of(.ptype = integer())),
    vctrs::new_list_of(rep(list(NULL), 10), ptype = integer())
  )
})

test_that("convert to vector works for map -> vctrs::list_of", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  values <- vctrs::list_of(
    data.frame(key = "key1", value = 1L),
    data.frame(key = c("key2", "key3"), value = c(2L, 3L)),
    NULL
  )

  array_list <- as_nanoarrow_array(
    arrow::Array$create(
      values,
      type = arrow::map_of(arrow::string(), arrow::int32())
    )
  )

  # Default conversion
  expect_identical(
    convert_array(array_list),
    values
  )
})

test_that("convert to vector works for fixed_size_list_of() -> matrix()", {
  mat <- matrix(1:6, ncol = 2, byrow = TRUE)
  array <- as_nanoarrow_array(mat)

  expect_identical(
    convert_array(array, matrix(double(), ncol = 2)),
    matrix(as.double(1:6), ncol = 2, byrow = TRUE)
  )
})

test_that("convert to vector errors for invalid matrix()", {
  expect_error(
    convert_array(as_nanoarrow_array(1:6), matrix()),
    "Can't convert array <int32> to R vector of type matrix"
  )

  mat <- matrix(1:6, ncol = 2, byrow = TRUE)
  array <- as_nanoarrow_array(mat)
  expect_error(
    convert_array(array, matrix(integer(), ncol = 3)),
    "Can't convert fixed_size_list(list_size=2) to matrix with 3 cols",
    fixed = TRUE
  )
})

test_that("convert to vector works for Date", {
  array_date <- as_nanoarrow_array(as.Date(c(NA, "2000-01-01")))
  expect_identical(
    convert_array(array_date),
    as.Date(c(NA, "2000-01-01"))
  )

  array_date <- as_nanoarrow_array(
    as.Date(c(NA, "2000-01-01")),
    schema = na_date64()
  )
  expect_identical(
    convert_array(array_date),
    as.POSIXct(c(NA, "2000-01-01"), tz = "UTC")
  )
})

test_that("convert to vector works for null -> Date", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, as.Date(character())),
    as.Date(rep(NA_character_, 10))
  )
})

test_that("convert to vector works for hms", {
  skip_if_not_installed("hms")

  array_time <- as_nanoarrow_array(hms::parse_hm("12:34"))
  expect_identical(
    convert_array(array_time),
    hms::parse_hm("12:34")
  )
})

test_that("convert to vector works for null -> hms", {
  skip_if_not_installed("hms")

  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, hms::hms()),
    hms::parse_hms(rep(NA_character_, 10))
  )
})

test_that("convert to vector works for POSIXct", {
  array_timestamp <- as_nanoarrow_array(
    as.POSIXct("2000-01-01 12:33", tz = "America/Halifax")
  )

  expect_identical(
    convert_array(array_timestamp),
    as.POSIXct("2000-01-01 12:33", tz = "America/Halifax")
  )
})

test_that("convert to vector works for null -> POSIXct", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, as.POSIXct(character(), tz = "America/Halifax")),
    as.POSIXct(rep(NA_character_, 10), tz = "America/Halifax")
  )
})

test_that("convert to vector works for difftime", {
  x <- as.difftime(123, units = "secs")
  array_duration <- as_nanoarrow_array(x)

  # default
  expect_identical(convert_array(array_duration), x)

  # explicit
  expect_identical(convert_array(array_duration, x), x)

  # explicit with other difftime units
  units(x) <- "mins"
  expect_identical(convert_array(array_duration, x), x)

  units(x) <- "hours"
  expect_identical(convert_array(array_duration, x), x)

  units(x) <- "days"
  expect_identical(convert_array(array_duration, x), x)

  units(x) <- "weeks"
  expect_equal(convert_array(array_duration, x), x)

  # with all Arrow units
  x <- as.difftime(123, units = "secs")
  array_duration <- as_nanoarrow_array(x, na_duration("s"))
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(x, na_duration("ms"))
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(x, na_duration("us"))
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(x, na_duration("ns"))
  expect_equal(convert_array(array_duration), x)

  # bad ptype values
  attr(x, "units") <- NULL
  expect_error(
    convert_array(array_duration, x),
    "Expected difftime 'units' attribute of type"
  )

  attr(x, "units") <- character()
  expect_error(
    convert_array(array_duration, x),
    "Expected difftime 'units' attribute of type"
  )

  attr(x, "units") <- integer(1)
  expect_error(
    convert_array(array_duration, x),
    "Expected difftime 'units' attribute of type"
  )

  attr(x, "units") <- "gazornenplat"
  expect_error(
    convert_array(array_duration, x),
    "Unexpected value for difftime 'units' attribute"
  )

  attr(x, "units") <- NA_character_
  expect_error(
    convert_array(array_duration, x),
    "Unexpected value for difftime 'units' attribute"
  )
})

test_that("convert to vector works for null -> difftime", {
  array <- nanoarrow_array_init(na_na())
  array$length <- 10
  array$null_count <- 10

  expect_identical(
    convert_array(array, as.difftime(numeric(), units = "secs")),
    as.difftime(rep(NA_real_, 10), units = "secs")
  )
})

test_that("convert to vector works for data frames nested inside lists", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  df_in_list <- vctrs::list_of(
    data.frame(x = 1:5),
    data.frame(x = 6:10),
    data.frame(x = 11:15)
  )

  nested_array <- as_nanoarrow_array(df_in_list)
  expect_identical(
    convert_array(nested_array),
    df_in_list
  )
})

test_that("convert to vector works for lists nested in data frames", {
  skip_if_not_installed("arrow")
  skip_if_not_installed("vctrs")

  df_in_list_in_df <- data.frame(
    x = vctrs::list_of(
      data.frame(x = 1:5),
      data.frame(x = 6:10),
      data.frame(x = 11:15)
    )
  )

  nested_array <- as_nanoarrow_array(df_in_list_in_df)
  expect_identical(
    convert_array(nested_array),
    df_in_list_in_df
  )
})
