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

test_that("vctrs extension type can roundtrip built-in vector types", {
  skip_if_not_installed("tibble")
  skip_if_not_installed("jsonlite")

  # Lists aren't automatically handled in nanoarrow conversion, so they
  # aren't listed here yet.
  vectors <- list(
    lgl = c(FALSE, TRUE, NA),
    int = c(0L, 1L, NA_integer_),
    dbl = c(0, 1, NA_real_),
    chr = c("a", NA_character_),
    posixct = as.POSIXct("2000-01-01 12:23", tz = "UTC"),
    posixlt = as.POSIXlt("2000-01-01 12:23", tz = "UTC"),
    date = as.Date("2000-01-01"),
    difftime = as.difftime(123, units = "secs"),
    data_frame_simple = data.frame(x = 1:5),
    data_frame_nested = tibble::tibble(x = 1:5, y = tibble::tibble(z = letters[1:5]))
  )

  for (nm in names(vectors)) {
    vctr <- vectors[[nm]]
    ptype <- vctrs::vec_ptype(vctr)
    schema <- na_vctrs(vctr)

    array <- as_nanoarrow_array(vctr, schema = schema)
    array_schema <- infer_nanoarrow_schema(array)

    # Roundtrip through convert_array()
    expect_true(nanoarrow_schema_identical(array_schema, schema))
    expect_identical(infer_nanoarrow_ptype(array), ptype)
    expect_identical(convert_array(array), vctr)

    # Roundtrip with an empty array stream
    stream <- basic_array_stream(list(), schema = schema)
    expect_identical(convert_array_stream(stream), ptype)

    # Roundtrip with multiple chunks
    stream <- basic_array_stream(list(array, array))
    expect_identical(convert_array_stream(stream), vctrs::vec_rep(vctr, 2))
  }
})

test_that("vctrs extension type respects `to` in convert_array()", {
  skip_if_not_installed("vctrs")

  vctr <- as.Date("2000-01-01")
  array <- as_nanoarrow_array(vctr, schema = na_vctrs(vctr))

  expect_identical(convert_array(array), vctr)
  expect_identical(
    convert_array(array, to = as.POSIXct(character())),
    vctrs::vec_cast(vctr, as.POSIXct(character()))
  )
})

test_that("serialize_ptype() can roundtrip R objects", {
  skip_if_not_installed("jsonlite")
  skip_if_not_installed("tibble")

  vectors <- list(
    null = NULL,
    raw = as.raw(c(0x00, 0x01, 0x02)),
    lgl = c(FALSE, TRUE, NA),
    int = c(0L, 1L, NA_integer_),
    dbl = c(0, 1, pi, NA_real_),
    chr = c("a", NA_character_),
    cmplx = c(complex(real = 1:3, imaginary = 3:1), NA_complex_),
    list = list(1, 2, x = 3, NULL),

    raw0 = raw(),
    lgl0 = logical(),
    int0 = integer(),
    dbl0 = double(),
    chr0 = character(),
    cmplx0 = complex(),
    list0 = list(),

    posixct = as.POSIXct("2000-01-01 12:23", tz = "UTC"),
    posixlt = as.POSIXlt("2000-01-01 12:23", tz = "UTC"),
    date = as.Date("2000-01-01"),
    difftime = as.difftime(123, units = "secs"),
    data_frame_simple = data.frame(x = 1:5),
    data_frame_nested = tibble::tibble(x = 1:5, y = tibble::tibble(z = letters[1:5]))
  )

  for (obj in vectors) {
    # Check that our serializer/deserializer can roundtrip
    expect_identical(
      unserialize_ptype(serialize_ptype(obj)),
      obj
    )

    # Check that our generated JSON is compatible with jsonlite's serde
    expect_identical(
      jsonlite::unserializeJSON(serialize_ptype(obj)),
      obj
    )

    expect_identical(
      unserialize_ptype(jsonlite::serializeJSON(obj, digits = 16)),
      obj
    )
  }
})

test_that("serialize_ptype() errors for unsupported R objects", {
  skip_if_not_installed("jsonlite")

  expect_error(
    serialize_ptype(quote(cat("I will eat you"))),
    "storage 'language' is not supported by serialize_ptype"
  )

  expect_error(
    unserialize_ptype(jsonlite::serializeJSON(quote(cat("I will eat you")))),
    "storage 'language' is not supported by unserialize_ptype"
  )

})
