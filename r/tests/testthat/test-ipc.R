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

test_that("read_nanoarrow() works for raw vectors", {
  stream <- read_nanoarrow(example_ipc_stream())
  expect_s3_class(stream, "nanoarrow_array_stream")
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for open connections", {
  con <- rawConnection(example_ipc_stream())
  on.exit(close(con))

  stream <- read_nanoarrow(con)
  expect_s3_class(stream, "nanoarrow_array_stream")
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for unopened connections", {
  tf <- tempfile()
  on.exit(unlink(tf))

  con <- file(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  con <- file(tf)
  # Don't close on exit, because we're supposed to do that

  stream <- read_nanoarrow(con)
  expect_true(isOpen(con))
  stream$release()
  expect_error(
    close(con),
    "invalid connection"
  )
})

test_that("read_nanoarrow() works for file paths", {
  tf <- tempfile()
  on.exit(unlink(tf))

  con <- file(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  stream <- read_nanoarrow(tf)
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for URLs", {
  tf <- tempfile()
  on.exit(unlink(tf))

  con <- file(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  stream <- read_nanoarrow(paste0("file://", tf))
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for compressed .gz file paths", {
  tf <- tempfile(fileext = ".gz")
  on.exit(unlink(tf))

  con <- gzfile(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  stream <- read_nanoarrow(tf)
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for compressed .bz2 file paths", {
  tf <- tempfile(fileext = ".bz2")
  on.exit(unlink(tf))

  con <- bzfile(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  stream <- read_nanoarrow(tf)
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() works for compressed .zip file paths", {
  tf <- tempfile(fileext = ".zip")
  tdir <- tempfile()
  on.exit(unlink(c(tf, tdir), recursive = TRUE))

  dir.create(tdir)
  uncompressed <- file.path(tdir, "file.arrows")
  con <- file(uncompressed, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  local({
    wd <- getwd()
    on.exit(setwd(wd))
    setwd(tdir)
    tryCatch(
      zip(tf, "file.arrows", extras = "-q"),
      error = function(...) skip("zip() not supported")
    )
  })

  stream <- read_nanoarrow(tf)
  expect_identical(
    as.data.frame(stream),
    data.frame(some_col = c(1L, 2L, 3L))
  )
})

test_that("read_nanoarrow() errors for compressed URL paths", {
  expect_error(
    read_nanoarrow("https://something.zip"),
    "Reading compressed streams from URLs"
  )
})

test_that("read_nanoarrow() errors for input with length != 1", {
  expect_error(
    read_nanoarrow(character(0)),
    "Can't interpret character"
  )
})

test_that("read_nanoarrow() errors zip archives that contain files != 1", {
  tf <- tempfile(fileext = ".zip")
  tdir <- tempfile()
  on.exit(unlink(c(tf, tdir), recursive = TRUE))

  dir.create(tdir)
  file.create(file.path(tdir, c("file1", "file2")))
  local({
    wd <- getwd()
    on.exit(setwd(wd))
    setwd(tdir)
    tryCatch(
      zip(tf, c("file1", "file2"), extras = "-q"),
      error = function(...) skip("zip() not supported")
    )
  })

  expect_error(
    read_nanoarrow(tf),
    "Unzip only supported of archives with exactly one file"
  )
})

test_that("read_nanoarrow() reports errors from readBin", {
  tf <- tempfile()
  on.exit(unlink(tf))
  writeLines("this is not a binary file", tf)

  con <- file(tf, open = "r")
  on.exit(close(con))

  expect_error(
    read_nanoarrow(con),
    "R execution error"
  )
})

test_that("read_nanoarrow() respects lazy argument", {
  expect_error(
    read_nanoarrow(raw(0), lazy = FALSE),
    "No data available on stream"
  )

  reader <- read_nanoarrow(raw(0), lazy = TRUE)
  expect_error(
    reader$get_next(),
    "No data available on stream"
  )

  tf <- tempfile()
  con <- rawConnection(raw(0))
  on.exit({
    close(con)
    unlink(tf)
  })

  expect_error(
    read_nanoarrow(con, lazy = FALSE),
    "No data available on stream"
  )

  reader <- read_nanoarrow(con, lazy = TRUE)
  expect_error(
    reader$get_next(),
    "No data available on stream"
  )

  file.create(tf)
  expect_error(
    read_nanoarrow(tf, lazy = FALSE),
    "No data available on stream"
  )

  reader <- read_nanoarrow(tf, lazy = TRUE)
  expect_error(
    reader$get_next(),
    "No data available on stream"
  )
})

test_that("read_nanoarrow() from connection errors when called from another thread", {
  skip_if_not_installed("arrow")
  skip_if_not(arrow::arrow_info()$capabilities["dataset"])
  skip_if_not_installed("dplyr")

  tf <- tempfile()
  tf_out <- tempfile()
  on.exit(unlink(c(tf, tf_out), recursive = TRUE))

  con <- file(tf, "wb")
  writeBin(example_ipc_stream(), con)
  close(con)

  stream <- read_nanoarrow(tf)
  reader <- arrow::as_record_batch_reader(stream)

  # There is an internal MakeSafeRecordBatchReader that ensures all read
  # calls happen on the R thread (used in DuckDB integration), but for now
  # this should at least error and not crash.
  expect_error(
    arrow::write_dataset(reader, tf_out),
    "Can't read from R connection on a non-R thread"
  )
})
