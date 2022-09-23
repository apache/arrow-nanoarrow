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

test_that("convert to vector works for partial_frame", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  expect_identical(
    from_nanoarrow_array(array, vctrs::partial_frame()),
    data.frame(a = 1L, b = "two")
  )
})

test_that("convert to vector works for data.frame", {
  array <- as_nanoarrow_array(data.frame(a = 1L, b = "two"))
  expect_identical(
    from_nanoarrow_array(array, data.frame(a = integer(), b = character())),
    data.frame(a = 1L, b = "two")
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
    "Can't convert array to character"
  )
})
