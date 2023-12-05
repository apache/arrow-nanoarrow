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

test_that("basic_array_stream() can create empty streams", {
  stream <- basic_array_stream(list(), na_int32())
  expect_identical(stream$get_schema()$format, "i")
  expect_null(stream$get_next())

  expect_error(
    basic_array_stream(list()),
    "Can't infer schema from first batch if there are zero batches"
  )
})

test_that("basic_array_stream() can create streams from batches", {
  stream <- basic_array_stream(
    list(
      data.frame(a = 1, b = "two", stringsAsFactors = FALSE),
      data.frame(a = 2, b = "three", stringsAsFactors = FALSE)
    )
  )

  expect_identical(stream$get_schema()$format, "+s")
  expect_identical(
    as.data.frame(stream$get_next()),
    data.frame(a = 1, b = "two", stringsAsFactors = FALSE)
  )
  expect_identical(
    as.data.frame(stream$get_next()),
    data.frame(a = 2, b = "three", stringsAsFactors = FALSE)
  )
  expect_null(stream$get_next())
})

test_that("basic_array_stream() can validate input or skip validation", {
  invalid_stream <- basic_array_stream(
    list(
      as_nanoarrow_array(1:5),
      as_nanoarrow_array(data.frame(a = 1:5))
    ),
    validate = FALSE
  )
  expect_s3_class(invalid_stream, "nanoarrow_array_stream")

  expect_error(
    basic_array_stream(
      list(
        as_nanoarrow_array(1:5),
        as_nanoarrow_array(data.frame(a = 1:5))
      ),
      validate = TRUE
    ),
    "Expected array with 2 buffer"
  )
})

test_that("nanoarrow_array_stream format, print, and str methods work", {
  array_stream <- as_nanoarrow_array_stream(data.frame(x = 1:10))
  expect_identical(format(array_stream), "<nanoarrow_array_stream struct<x: int32>>")
  expect_output(expect_identical(str(array_stream), array_stream), "nanoarrow_array_stream")
  expect_output(expect_identical(print(array_stream), array_stream), "nanoarrow_array_stream")
})

test_that("released nanoarrow_array_stream format, print, and str methods work", {
  array_stream <- nanoarrow_allocate_array_stream()
  expect_identical(format(array_stream), "<nanoarrow_array_stream[invalid pointer]>")
  expect_output(expect_identical(str(array_stream), array_stream), "nanoarrow_array_stream")
  expect_output(expect_identical(print(array_stream), array_stream), "nanoarrow_array_stream")
})

test_that("as_nanoarrow_array_stream() works for nanoarow_array_stream", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(as_nanoarrow_array_stream(stream), stream)

  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(
    as_nanoarrow_array_stream(stream, schema = na_struct(list(x = na_int32()))),
    stream
  )

  skip_if_not_installed("arrow")
  expect_snapshot_error(
    as_nanoarrow_array_stream(stream, schema = na_struct(list(x = na_double())))
  )
})

test_that("as_nanoarrow_array_stream() works for nanoarow_array", {
  array <- as_nanoarrow_array(data.frame(x = 1:5))

  stream <- as_nanoarrow_array_stream(array)
  expect_identical(infer_nanoarrow_schema(stream)$format, "+s")
  expect_identical(
    lapply(collect_array_stream(stream), as.data.frame),
    list(data.frame(x = 1:5))
  )

  # With explicit but identical schema
  stream <- as_nanoarrow_array_stream(array, schema = na_struct(list(x = na_int32())))
  expect_identical(infer_nanoarrow_schema(stream)$format, "+s")
  expect_identical(
    lapply(collect_array_stream(stream), as.data.frame),
    list(data.frame(x = 1:5))
  )

  # With schema requiring a cast (not implemented in arrow)
  skip_if_not_installed("arrow")
  expect_snapshot_error(
    as_nanoarrow_array_stream(array, schema = na_struct(list(x = na_double())))
  )
})

test_that("infer_nanoarrow_schema() is implemented for streams", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  schema <- infer_nanoarrow_schema(stream)
  expect_identical(schema$children$x$format, "i")
})

test_that("as.data.frame() is implemented for streams", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(
    as.data.frame(stream),
    data.frame(x = 1:5)
  )
  expect_false(nanoarrow_pointer_is_valid(stream))
})

test_that("as.vector() is implemented for streams", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(
    as.vector(stream),
    data.frame(x = 1:5)
  )
  expect_false(nanoarrow_pointer_is_valid(stream))
})

test_that("nanoarrow_array_stream list interface works", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(length(stream), 3L)
  expect_identical(names(stream), c("get_schema", "get_next", "release"))
  expect_identical(formals(stream[["get_schema"]]), formals(stream$get_schema))
  expect_identical(formals(stream[["get_next"]]), formals(stream$get_next))
  expect_identical(formals(stream[["release"]]), formals(stream$release))
  expect_null(stream[["this key does not exist"]])
})

test_that("nanoarrow_array_stream can get_schema() and get_next()", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_identical(stream$get_schema()$format, "+s")
  expect_identical(as.data.frame(stream$get_next()), data.frame(x = 1:5))
  expect_null(stream$get_next())
})

test_that("nanoarrow_array_stream can release()", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_true(nanoarrow_pointer_is_valid(stream))
  stream$release()
  expect_false(nanoarrow_pointer_is_valid(stream))
})

test_that("nanoarrow_array_stream can validate or not on get_next()", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_error(
    stream$get_next(schema = na_int32()),
    "Expected array with 2 buffer"
  )

  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  expect_silent(
    stream$get_next(
      schema = na_int32(),
      validate = FALSE
    )
  )
})

test_that("nanoarrow_array_stream get_next() with schema = NULL", {
  stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
  array <- stream$get_next(schema = NULL)
  expect_error(infer_nanoarrow_schema(array), "has no associated schema")
})

test_that("User array stream finalizers are run on explicit release", {
  stream <- basic_array_stream(list(1:5))
  stream <- array_stream_set_finalizer(stream, function() cat("All done!"))
  expect_output(stream$release(), "All done!")
  expect_silent(stream$release())
})

test_that("User array stream finalizers are run on explicit release even when moved", {
  stream <- basic_array_stream(list(1:5))
  stream <- array_stream_set_finalizer(stream, function() cat("All done!"))

  stream2 <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_move(stream, stream2)
  expect_false(nanoarrow_pointer_is_valid(stream))
  expect_silent(nanoarrow_pointer_release(stream))
  expect_output(stream2$release(), "All done!")
  expect_silent(stream2$release())
})

test_that("User array stream finalizers are run on explicit release even when exported", {
  stream <- basic_array_stream(list(1:5))
  stream <- array_stream_set_finalizer(stream, function() cat("All done!"))

  stream2 <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_export(stream, stream2)
  expect_false(nanoarrow_pointer_is_valid(stream))
  expect_silent(nanoarrow_pointer_release(stream))
  expect_output(stream2$release(), "All done!")
  expect_silent(stream2$release())
})

test_that("Errors from user array stream finalizer are ignored", {
  stream <- basic_array_stream(list(1:5))
  stream <- array_stream_set_finalizer(stream, function() stop("Error that will be ignored"))
  # Because this comes from REprintf(), it's not a message and not "output"
  # according to testthat, so we use capture.output()
  expect_identical(
    capture.output(stream$release(), type = "message"),
    "Error evaluating user-supplied array stream finalizer"
  )

  expect_false(nanoarrow_pointer_is_valid(stream))
  expect_silent(stream$release())
})
