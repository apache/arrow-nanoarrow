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

test_that("read_array_stream() works for raw vectors", {
  stream <- read_array_stream(example_ipc_stream())
  expect_s3_class(stream, "nanoarrow_array_stream")
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_array_stream() works for open connections", {
  con <- rawConnection(example_ipc_stream())
  on.exit(close(con))

  stream <- read_array_stream(con)
  expect_s3_class(stream, "nanoarrow_array_stream")
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_array_stream() works for unopened connections", {
  tf <- tempfile()
  on.exit(unlink(tf))

  con <- file(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  con <- file(tf)
  # Don't close on exit, because we're supposed to do that

  stream <- read_array_stream(con)
  expect_true(isOpen(con))
  stream$release()
  expect_error(
    close(con),
    "invalid connection"
  )
})

test_that("read_array_stream() reports errors from readBin", {
  tf <- tempfile()
  on.exit(unlink(tf))
  writeLines("this is not a binary file", tf)

  con <- file(tf, open = "r")
  on.exit(close(con))

  stream <- read_array_stream(con)
  expect_error(
    stream$get_next(),
    "can only read from a binary connection"
  )
})
