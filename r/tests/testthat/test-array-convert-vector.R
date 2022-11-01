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

test_that("infer_nanoarrow_ptype() works for basic types", {
  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(vctrs::unspecified())),
    vctrs::unspecified()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(logical())),
    logical()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(integer())),
    integer()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(double())),
    double()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(character())),
    character()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(data.frame(x = character()))),
    data.frame(x = character())
  )
})

test_that("infer_nanoarrow_ptype() errors for types it can't infer",  {
  unsupported_array <- arrow::concat_arrays(type = arrow::decimal256(3, 4))
  expect_error(
    infer_nanoarrow_ptype(as_nanoarrow_array(unsupported_array)),
    "Can't infer R vector type for array <d:3,4,256>"
  )

  unsupported_struct <- arrow::concat_arrays(
    type = arrow::struct(col = arrow::decimal256(3, 4))
  )
  expect_error(
    infer_nanoarrow_ptype(as_nanoarrow_array(unsupported_struct)),
    "Can't infer R vector type for `col` <d:3,4,256>"
  )
})

test_that("from_nanoarrow_array() errors for invalid arrays", {
  array <- as_nanoarrow_array(1:10)
  nanoarrow_array_set_schema(
    array,
    infer_nanoarrow_schema("chr"),
    validate = FALSE
  )

  expect_error(
    from_nanoarrow_array(array),
    "Expected array with 3 buffer"
  )
})

test_that("from_nanoarrow_array() errors for unsupported ptype", {
  array <- as_nanoarrow_array(1:10)

  # an S3 unsupported type
  expect_error(
    from_nanoarrow_array(array, structure(list(), class = "some_class")),
    "Can't convert array <i> to R vector of type some_class"
  )

  # A non-S3 unsupported type
  expect_error(
    from_nanoarrow_array(array, environment()),
    "Can't convert array <i> to R vector of type environment"
  )
})

test_that("from_nanoarrow_array() errors for unsupported array", {
  unsupported_array <- arrow::concat_arrays(type = arrow::decimal256(3, 4))
  expect_error(
    from_nanoarrow_array(as_nanoarrow_array(unsupported_array)),
    "Can't infer R vector type for array <d:3,4,256>"
  )
})

test_that("convert to vector works for data.frame", {
  df <- data.frame(a = 1L, b = "two", c = 3, d = TRUE)
  array <- as_nanoarrow_array(df)

  expect_identical(from_nanoarrow_array(array, NULL), df)
  expect_identical(from_nanoarrow_array(array, df), df)

  expect_error(
    from_nanoarrow_array(array, data.frame(a = integer(), b = raw())),
    "Expected data.frame\\(\\) ptype with 4 column\\(s\\) but found 2 column\\(s\\)"
  )

  bad_ptype <- data.frame(a = integer(), b = raw(), c = double(), d = integer())
  expect_error(
    from_nanoarrow_array(array, bad_ptype),
    "Can't convert `b` <u> to R vector of type raw"
  )
})

test_that("convert to vector works for partial_frame", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  expect_identical(
    from_nanoarrow_array(array, vctrs::partial_frame()),
    data.frame(a = 1L, b = "two")
  )
})

test_that("convert to vector works for tibble", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  expect_identical(
    from_nanoarrow_array(array, tibble::tibble(a = integer(), b = character())),
    tibble::tibble(a = 1L, b = "two")
  )

  # Check nested tibble at both levels
  tbl_nested_df <- tibble::tibble(a = 1L, b = "two", c = data.frame(a = 3))
  array_nested <- as_nanoarrow_array(tbl_nested_df)

  expect_identical(
    from_nanoarrow_array(array_nested, tbl_nested_df),
    tbl_nested_df
  )

  df_nested_tbl <- as.data.frame(tbl_nested_df)
  df_nested_tbl$c <- tibble::as_tibble(df_nested_tbl$c)
  expect_identical(
    from_nanoarrow_array(array_nested, df_nested_tbl),
    df_nested_tbl
  )
})

test_that("convert to vector works for unspecified()", {
  array <- as_nanoarrow_array(
    arrow::Array$create(rep(NA, 10), arrow::null())
  )

  # implicit for null type
  expect_identical(
    from_nanoarrow_array(array, to = NULL),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for null type
  expect_identical(
    from_nanoarrow_array(array, vctrs::unspecified()),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for non-null type that is all NAs
  array <- as_nanoarrow_array(rep(NA_integer_, 10))
  expect_identical(
    from_nanoarrow_array(array, vctrs::unspecified()),
    vctrs::vec_cast(rep(NA, 10), vctrs::unspecified())
  )

  # explicit for non-null type that is not all NAs
  array <- as_nanoarrow_array(c(1L, rep(NA_integer_, 9)))
  expect_warning(
    expect_identical(
      from_nanoarrow_array(array, vctrs::unspecified()),
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
      from_nanoarrow_array(
        as_nanoarrow_array(vals, schema = arrow_numeric_types[[!!nm]]),
        logical()
      ),
      vals != 0
    )
  }

  vals_no_na <- 0:10
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      from_nanoarrow_array(
        as_nanoarrow_array(vals_no_na, schema = arrow_numeric_types[[!!nm]]),
        logical()
      ),
      vals_no_na != 0
    )
  }

  # Boolean array to logical
  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      logical()
    ),
    c(NA, TRUE, FALSE)
  )

  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      logical()
    ),
    c(TRUE, FALSE)
  )
})

test_that("convert to vector errors for bad array to logical()", {
  expect_error(
    from_nanoarrow_array(as_nanoarrow_array(letters), logical()),
    "Can't convert array <u> to R vector of type logical"
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
      from_nanoarrow_array(
        as_nanoarrow_array(ints, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints
    )
  }

  ints_no_na <- 0:10
  for (nm in names(arrow_int_types)) {
    expect_identical(
      from_nanoarrow_array(
        as_nanoarrow_array(ints_no_na, schema = arrow_int_types[[!!nm]]),
        integer()
      ),
      ints_no_na
    )
  }

  # Boolean array to integer
  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(NA, 1L, 0L)
  )

  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      integer()
    ),
    c(1L, 0L)
  )
})

test_that("convert to vector warns for invalid integer()", {
  array <- as_nanoarrow_array(arrow::as_arrow_array(.Machine$double.xmax))
  expect_warning(
    expect_identical(from_nanoarrow_array(array, integer()), NA_integer_),
    "1 value\\(s\\) outside integer range set to NA"
  )

  array <- as_nanoarrow_array(arrow::as_arrow_array(c(NA, .Machine$double.xmax)))
  expect_warning(
    expect_identical(from_nanoarrow_array(array, integer()), c(NA_integer_, NA_integer_)),
    "1 value\\(s\\) outside integer range set to NA"
  )
})

test_that("convert to vector errors for bad array to integer()", {
  expect_error(
    from_nanoarrow_array(as_nanoarrow_array(letters), integer()),
    "Can't convert array <u> to R vector of type integer"
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
      from_nanoarrow_array(
        as_nanoarrow_array(vals, schema = arrow_numeric_types[[!!nm]]),
        double()
      ),
      vals
    )
  }

  vals_no_na <- as.double(0:10)
  for (nm in names(arrow_numeric_types)) {
    expect_identical(
      from_nanoarrow_array(
        as_nanoarrow_array(vals_no_na, schema = arrow_numeric_types[[!!nm]]),
        double()
      ),
      vals_no_na
    )
  }

  # Boolean array to double
  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(NA, TRUE, FALSE), schema = arrow::boolean()),
      double()
    ),
    as.double(c(NA, 1L, 0L))
  )

  expect_identical(
    from_nanoarrow_array(
      as_nanoarrow_array(c(TRUE, FALSE), schema = arrow::boolean()),
      double()
    ),
    as.double(c(1L, 0L))
  )
})

test_that("convert to vector errors for bad array to double()", {
  expect_error(
    from_nanoarrow_array(as_nanoarrow_array(letters), double()),
    "Can't convert array <u> to R vector of type numeric"
  )
})

test_that("convert to vector works for character()", {
  array <- as_nanoarrow_array(letters)
  expect_identical(
    from_nanoarrow_array(array, character()),
    letters
  )

  # make sure we get altrep here
  expect_true(is_nanoarrow_altrep(from_nanoarrow_array(array, character())))

  # check an array that we can't convert
  expect_error(
    from_nanoarrow_array(as_nanoarrow_array(1:5), character()),
    "Can't convert array <i> to R vector of type character"
  )
})

test_that("convert to vector works for character()", {
  array <- as_nanoarrow_array(list(as.raw(1:5)), schema = arrow::binary())
  expect_identical(
    from_nanoarrow_array(array),
    list(as.raw(1:5))
  )

  expect_identical(
    from_nanoarrow_array(array, list()),
    list(as.raw(1:5))
  )
})
