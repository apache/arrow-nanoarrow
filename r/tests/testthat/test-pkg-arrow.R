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

test_that("array to Array works", {
  skip_if_not_installed("arrow")

  int <- arrow::as_arrow_array(as_nanoarrow_array(1:5))
  expect_true(int$Equals(arrow::Array$create(1:5)))

  dbl <- arrow::as_arrow_array(as_nanoarrow_array(1:5, schema = arrow::float64()))
  expect_true(dbl$Equals(arrow::Array$create(1:5, type = arrow::float64())))

  dbl_casted <- arrow::as_arrow_array(as_nanoarrow_array(1:5), type = arrow::float64())
  expect_true(dbl$Equals(arrow::Array$create(1:5, type = arrow::float64())))

  chr <- arrow::as_arrow_array(as_nanoarrow_array(c("one", "two")))
  expect_true(chr$Equals(arrow::Array$create(c("one", "two"))))
})

test_that("array to RecordBatch works", {
  skip_if_not_installed("arrow")

  df <- data.frame(a = 1:5, b = letters[1:5])
  batch <- arrow::as_record_batch(as_nanoarrow_array(df))
  expect_true(
    batch$Equals(arrow::record_batch(a = 1:5, b = letters[1:5]))
  )

  batch_casted <- arrow::as_record_batch(
    as_nanoarrow_array(df),
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_true(
    batch_casted$Equals(
      arrow::record_batch(a = as.double(1:5), b = letters[1:5])
    )
  )
})

test_that("array to Table works", {
  skip_if_not_installed("arrow")

  df <- data.frame(a = 1:5, b = letters[1:5])
  table <- arrow::as_arrow_table(as_nanoarrow_array(df))
  expect_true(
    table$Equals(arrow::arrow_table(a = 1:5, b = letters[1:5]))
  )

  table_casted <- arrow::as_arrow_table(
    as_nanoarrow_array(df),
    schema = arrow::schema(a = arrow::float64(), b = arrow::string())
  )
  expect_true(
    table_casted$Equals(
      arrow::arrow_table(a = as.double(1:5), b = letters[1:5])
    )
  )
})
