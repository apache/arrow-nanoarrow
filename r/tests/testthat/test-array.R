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

test_that("as_nanoarrow_array() / from_nanoarrow_array() default method works", {
  array <- as_nanoarrow_array(1:10)
  expect_identical(from_nanoarrow_array(array), 1:10)

  array <- as_nanoarrow_array(1:10, schema = arrow::float64())
  expect_identical(from_nanoarrow_array(array), as.double(1:10))
})

test_that("infer_nanoarrow_schema() works for nanoarrow_array", {
  array <- as_nanoarrow_array(1:10)
  schema <- infer_nanoarrow_schema(array)
  expect_true(arrow::as_data_type(schema)$Equals(arrow::int32()))

  nanoarrow_array_set_schema(array, NULL)
  expect_error(infer_nanoarrow_schema(array), "has no associated schema")
})

test_that("nanoarrow_array_set_schema() errors for invalid schema/array", {
  array <- as_nanoarrow_array(integer())
  schema <- infer_nanoarrow_schema(character())
  expect_error(
    nanoarrow_array_set_schema(array, schema),
    "Expected array with 3 buffer\\(s\\) but found 2 buffer\\(s\\)"
  )
})
