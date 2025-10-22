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

test_that("as_nanoarrow_schema() for Python object", {
  skip_if_not(has_reticulate_with_nanoarrow())

  na <- reticulate::import("nanoarrow")

  expect_identical(as_nanoarrow_schema(na$binary())$format, "z")
})

test_that("as_nanoarrow_array() for Python object", {
  skip_if_not(has_reticulate_with_nanoarrow())

  na <- reticulate::import("nanoarrow")

  array <- as_nanoarrow_array(na$Array(1:5, na_int32()))
  expect_identical(
    convert_array(array),
    1:5
  )

  # Check schema request argument
  lst <- reticulate::py_eval('[1, 2, 3, 4, 5]', convert = FALSE)
  array <- as_nanoarrow_array(lst, schema = na_int32())
  expect_identical(
    convert_array(array),
    1:5
  )
})

test_that("as_nanoarrow_stream() for Python object", {
  skip_if_not(has_reticulate_with_nanoarrow())

  na <- reticulate::import("nanoarrow")

  stream <- as_nanoarrow_array_stream(na$ArrayStream(1:5, na_int32()))
  expect_identical(
    convert_array_stream(stream),
    1:5
  )

  # Check schema request argument
  lst <- reticulate::py_eval('[1, 2, 3, 4, 5]', convert = FALSE)
  stream <- as_nanoarrow_array_stream(lst, schema = na_int32())
  expect_identical(
    convert_array_stream(stream),
    1:5
  )
})

test_that("schemas can be converted to Python and back", {
  skip_if_not(has_reticulate_with_nanoarrow())

  py_schema <- reticulate::r_to_py(na_binary())
  expect_s3_class(py_schema, "nanoarrow.schema.Schema")
  r_schema <- reticulate::py_to_r(py_schema)
  expect_identical(r_schema$format, "z")
})

test_that("arrays can be converted to Python and back", {
  skip_if_not(has_reticulate_with_nanoarrow())

  py_array <- reticulate::r_to_py(as_nanoarrow_array(1:5))
  expect_s3_class(py_array, "nanoarrow.array.Array")
  expect_identical(reticulate::py_to_r(py_array$to_pylist()), 1:5)
  r_array <- reticulate::py_to_r(py_array)
  expect_identical(convert_array(r_array), 1:5)
})
