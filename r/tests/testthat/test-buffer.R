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

test_that("as_nanoarrow_buffer() works for nanoarrow_buffer", {
  buffer <- as_nanoarrow_array(1:4)$buffers[[2]]
  expect_identical(as_nanoarrow_buffer(buffer), buffer)
})

test_that("as_nanoarrow_buffer() works for R atomic types", {
  skip_if_not_installed("blob")

  buffer_null <- as_nanoarrow_buffer(NULL)
  expect_identical(as.raw(buffer_null), raw(0))
  expect_identical(convert_buffer(buffer_null), blob::blob(raw(0)))
  expect_identical(as.vector(buffer_null), convert_buffer(buffer_null))

  buffer_raw <- as_nanoarrow_buffer(as.raw(0x00))
  expect_identical(as.raw(buffer_raw), raw(1))
  expect_identical(convert_buffer(buffer_raw), blob::blob(as.raw(0x00)))

  buffer_lgl <- as_nanoarrow_buffer(FALSE)
  expect_identical(as.raw(buffer_lgl), raw(4))
  expect_identical(convert_buffer(buffer_lgl), 0L)

  buffer_int <- as_nanoarrow_buffer(0L)
  expect_identical(as.raw(buffer_lgl), raw(4))
  expect_identical(convert_buffer(buffer_lgl), 0L)

  buffer_dbl <- as_nanoarrow_buffer(0)
  expect_identical(as.raw(buffer_lgl), raw(4))

  buffer_cplx <- as_nanoarrow_buffer(complex(real = 0, imaginary = 0))
  expect_identical(as.raw(buffer_cplx), raw(16))
  expect_identical(convert_buffer(buffer_cplx), c(0, 0))

  buffer_chr <- as_nanoarrow_buffer("1234")
  expect_identical(as.raw(buffer_chr), charToRaw("1234"))
  expect_identical(convert_buffer(buffer_chr), "1234")
})

test_that("buffers can be printed", {
  expect_snapshot(str(as_nanoarrow_buffer(1:10)))
  expect_snapshot(str(as_nanoarrow_buffer(1:10000)))
  expect_snapshot(str(as_nanoarrow_buffer(strrep("abcdefg", 100))))
  expect_snapshot(str(as_nanoarrow_buffer(charToRaw(strrep("abcdefg", 100)))))

  array <- as_nanoarrow_array(1:100)
  nanoarrow_array_set_schema(array, NULL)
  expect_snapshot(str(array$buffers[[2]]))
})

test_that("buffers whose pointer is NULL print as such", {
  empty_buffer <- as_nanoarrow_buffer(logical())
  expect_match(format(empty_buffer), "data<int32>[null]", fixed = TRUE)
})

test_that("as_nanoarrow_buffer() errors for unsupported types", {
  expect_error(
    as_nanoarrow_buffer(NA_character_),
    "NA_character_ not supported"
  )

  expect_error(
    as_nanoarrow_buffer(environment()),
    "Can't convert object of type environment to nanoarrow_buffer"
  )
})
