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
    infer_nanoarrow_schema("chr"),
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
  unsupported_array <- arrow::concat_arrays(type = arrow::decimal256(3, 4))
  expect_error(
    convert_array(as_nanoarrow_array(unsupported_array)),
    "Can't infer R vector type for array <decimal256\\(3, 4\\)>"
  )
})

test_that("convert to vector works for data.frame", {
  df <- data.frame(a = 1L, b = "two", c = 3, d = TRUE)
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

test_that("convert to vector works for partial_frame", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  expect_identical(
    convert_array(array, vctrs::partial_frame()),
    data.frame(a = 1L, b = "two")
  )
})

test_that("convert to vector works for function()", {
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
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
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

test_that("convert to vector works for struct-style vectors", {
  array <- as_nanoarrow_array(as.POSIXlt("2021-01-01", tz = "America/Halifax"))
  expect_identical(
    convert_array(array),
    as.data.frame(unclass(as.POSIXlt("2021-01-01", tz = "America/Halifax")))
  )

  array <- as_nanoarrow_array(as.POSIXlt("2021-01-01", tz = "America/Halifax"))
  expect_identical(
    convert_array(array, as.POSIXlt("2021-01-01", tz = "America/Halifax")),
    as.POSIXlt("2021-01-01", tz = "America/Halifax")
  )
})

test_that("convert to vector works for unspecified()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))

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
    "1 non-null value\\(s\\) set to NA"
  )
})

test_that("convert to vector works for valid logical()", {
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
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, logical()),
    rep(NA, 10)
  )
})

test_that("convert to vector errors for bad array to logical()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), logical()),
    "Can't convert array <string> to R vector of type logical"
  )
})

test_that("convert to vector works for valid integer()", {
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

test_that("convert to vector works for null -> logical()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, integer()),
    rep(NA_integer_, 10)
  )
})

test_that("convert to vector warns for invalid integer()", {
  array <- as_nanoarrow_array(arrow::as_arrow_array(.Machine$double.xmax))
  expect_warning(
    expect_identical(convert_array(array, integer()), NA_integer_),
    "1 value\\(s\\) outside integer range set to NA"
  )

  array <- as_nanoarrow_array(arrow::as_arrow_array(c(NA, .Machine$double.xmax)))
  expect_warning(
    expect_identical(convert_array(array, integer()), c(NA_integer_, NA_integer_)),
    "1 value\\(s\\) outside integer range set to NA"
  )
})

test_that("convert to vector errors for bad array to integer()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), integer()),
    "Can't convert array <string> to R vector of type integer"
  )
})

test_that("convert to vector works for valid double()", {
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

test_that("convert to vector works for decimal128 -> double()", {
  array <- as_nanoarrow_array(arrow::Array$create(1:10)$cast(arrow::decimal128(20, 10)))
  expect_equal(
    convert_array(array, double()),
    as.double(1:10)
  )
})

test_that("convert to vector works for null -> double()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, double()),
    rep(NA_real_, 10)
  )
})

test_that("convert to vector errors for bad array to double()", {
  expect_error(
    convert_array(as_nanoarrow_array(letters), double()),
    "Can't convert array <string> to R vector of type numeric"
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
    convert_array(as_nanoarrow_array(1:5), character()),
    "Can't convert array <int32> to R vector of type character"
  )
})

test_that("convert to vector works for null -> character()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  all_nulls <- convert_array(array, character())
  nanoarrow_altrep_force_materialize(all_nulls)
  expect_identical(
    all_nulls,
    rep(NA_character_, 10)
  )
})

test_that("convert to vector works for blob::blob()", {
  array <- as_nanoarrow_array(list(as.raw(1:5)), schema = arrow::binary())
  expect_identical(
    convert_array(array),
    blob::blob(as.raw(1:5))
  )

  expect_identical(
    convert_array(array, blob::blob()),
    blob::blob(as.raw(1:5))
  )
})

test_that("convert to vector works for null -> blob::blob()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, blob::blob()),
    blob::new_blob(rep(list(NULL), 10))
  )
})

test_that("convert to vector works for list -> vctrs::list_of", {
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
    convert_array(array_list, vctrs::list_of(.ptype = character())),
    "Can't convert array"
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
    convert_array(array_list, vctrs::list_of(.ptype = character())),
    "Can't convert array"
  )
})

test_that("convert to vector works for fixed_size_list -> vctrs::list_of", {
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
    convert_array(array_list, vctrs::list_of(.ptype = character())),
    "Can't convert array"
  )
})

test_that("convert to vector works for null -> vctrs::list_of()", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, vctrs::list_of(.ptype = integer())),
    vctrs::new_list_of(rep(list(NULL), 10), ptype = integer())
  )
})

test_that("convert to vector works for Date", {
  array_date <- as_nanoarrow_array(as.Date(c(NA, "2000-01-01")))
  expect_identical(
    convert_array(array_date),
    as.Date(c(NA, "2000-01-01"))
  )

  array_date <- as_nanoarrow_array(
    arrow::Array$create(as.Date(c(NA, "2000-01-01")), arrow::date64())
  )
  expect_identical(
    convert_array(array_date),
    as.POSIXct(c(NA, "2000-01-01"), tz = "UTC")
  )
})

test_that("convert to vector works for null -> Date", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, as.Date(character())),
    as.Date(rep(NA_character_, 10))
  )
})

test_that("convert to vector works for hms", {
  array_time <- as_nanoarrow_array(hms::parse_hm("12:34"))
  expect_identical(
    convert_array(array_time),
    hms::parse_hm("12:34")
  )
})

test_that("convert to vector works for null -> hms", {
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
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
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
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
  array_duration <- as_nanoarrow_array(
    arrow::Array$create(x, arrow::duration("s"))
  )
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(
    arrow::Array$create(x, arrow::duration("ms"))
  )
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(
    arrow::Array$create(x, arrow::duration("us"))
  )
  expect_identical(convert_array(array_duration), x)

  array_duration <- as_nanoarrow_array(
    arrow::Array$create(x, arrow::duration("ns"))
  )
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
  array <- as_nanoarrow_array(arrow::Array$create(rep(NA, 10), arrow::null()))
  expect_identical(
    convert_array(array, as.difftime(numeric(), units = "secs")),
    as.difftime(rep(NA_real_, 10), units = "secs")
  )
})

test_that("convert to vector works for data frames nested inside lists", {
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

test_that("convert to vector warns for stripped extension type", {
  ext_arr <- as_nanoarrow_array(
    arrow::Array$create(vctrs::new_vctr(1:5, class = "my_vctr"))
  )
  expect_warning(
    expect_identical(convert_array(ext_arr), 1:5),
    "Converting unknown extension arrow.r.vctrs"
  )

  nested_ext_array <- as_nanoarrow_array(
    arrow::record_batch(
      x = vctrs::new_vctr(1:5, class = "my_vctr")
    )
  )
  expect_warning(
    expect_identical(convert_array(nested_ext_array), data.frame(x = 1:5)),
    "x: Converting unknown extension arrow.r.vctrs"
  )
})

test_that("convert to vector errors for dictionary types", {
  dict_array <- as_nanoarrow_array(factor(letters[1:5]))
  expect_error(
    convert_array(dict_array, character()),
    "Conversion to dictionary-encoded array is not supported"
  )
})
