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

test_that("infer_nanoarrow_ptype() works on arrays, schemas, and streams", {
  array <- as_nanoarrow_array(logical())
  expect_identical(infer_nanoarrow_ptype(array), logical())

  schema <- infer_nanoarrow_schema(array)
  expect_identical(infer_nanoarrow_ptype(schema), logical())

  stream <- as_nanoarrow_array_stream(data.frame(x = logical()))
  expect_identical(infer_nanoarrow_ptype(stream), data.frame(x = logical()))

  expect_error(
    infer_nanoarrow_ptype("not valid"),
    "must be a nanoarrow_schema"
  )
})

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
    infer_nanoarrow_ptype(as_nanoarrow_schema(na_decimal128(2, 3))),
    double()
  )

  expect_identical(
    infer_nanoarrow_ptype(as_nanoarrow_array(character())),
    character()
  )

  expect_identical(
    infer_nanoarrow_ptype(
      as_nanoarrow_array(data.frame(x = character(), stringsAsFactors = FALSE))
    ),
    data.frame(x = character(), stringsAsFactors = FALSE)
  )
})

test_that("infer_nanoarrow_ptype() infers ptypes for date/time types", {
  array_date <- as_nanoarrow_array(as.Date("2000-01-01"))
  expect_identical(
    infer_nanoarrow_ptype(array_date),
    as.Date(character())
  )

  array_time <- as_nanoarrow_array(hms::parse_hm("12:34"))
  expect_identical(
    infer_nanoarrow_ptype(array_time),
    hms::hms()
  )

  array_duration <- as_nanoarrow_array(as.difftime(123, units = "secs"))
  expect_identical(
    infer_nanoarrow_ptype(array_duration),
    as.difftime(numeric(), units = "secs")
  )

  array_timestamp <- as_nanoarrow_array(
    as.POSIXct("2000-01-01 12:33", tz = "America/Halifax")
  )
  expect_identical(
    infer_nanoarrow_ptype(array_timestamp),
    as.POSIXct(character(), tz = "America/Halifax")
  )
})

test_that("infer_nanoarrow_ptype() infers ptypes for nested types", {
  skip_if_not_installed("arrow")

  array_list <- as_nanoarrow_array(vctrs::list_of(integer()))
  expect_identical(
    infer_nanoarrow_ptype(array_list),
    vctrs::list_of(.ptype = integer())
  )

  array_fixed_size <- as_nanoarrow_array(
    arrow::Array$create(
      list(1:5),
      arrow::fixed_size_list_of(arrow::int32(), 5)
    )
  )
  expect_identical(
    infer_nanoarrow_ptype(array_fixed_size),
    vctrs::list_of(.ptype = integer())
  )
})

test_that("infer_nanoarrow_ptype() errors for types it can't infer",  {
  unsupported_array <- nanoarrow_array_init(na_list_view(na_int32()))
  expect_error(
    infer_nanoarrow_ptype(as_nanoarrow_array(unsupported_array)),
    "Can't infer R vector type for <list_view<item: int32>>"
  )

  unsupported_struct <- nanoarrow_array_init(
    na_struct(list(col = na_list_view(na_int32())))
  )
  expect_error(
    infer_nanoarrow_ptype(as_nanoarrow_array(unsupported_struct)),
    "Can't infer R vector type for `col` <list_view<item: int32>>"
  )
})
