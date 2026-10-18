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

test_that("function_array_stream() lazily creates arrays and reaches EOF", {
  calls <- 0L
  stream <- function_array_stream(
    function() {
      calls <<- calls + 1L
      if (calls > 2L) NULL else data.frame(x = calls)
    },
    na_struct(list(x = na_int32()))
  )

  expect_identical(calls, 0L)
  expect_identical(stream$get_schema()$format, "+s")
  expect_identical(calls, 0L)
  expect_identical(as.data.frame(stream$get_next()), data.frame(x = 1L))
  expect_identical(as.data.frame(stream$get_next()), data.frame(x = 2L))
  expect_null(stream$get_next())
  expect_null(stream$get_next())
  expect_identical(calls, 3L)
  stream$release()
})

test_that("function_array_stream() owns a schema copy", {
  schema <- na_int32()
  stream <- function_array_stream(function() 42L, schema)
  nanoarrow_pointer_release(schema)

  expect_identical(stream$get_schema()$format, "i")
  expect_identical(stream$get_next()$length, 1L)
  stream$release()
})

test_that("function_array_stream() accepts NULL immediately and zero-row arrays", {
  calls <- 0L
  empty <- function_array_stream(
    function() {
      calls <<- calls + 1L
      NULL
    }, na_int32()
  )
  expect_null(empty$get_next())
  expect_null(empty$get_next())
  expect_identical(calls, 1L)
  empty$release()

  stream <- function_array_stream(
    function() data.frame(x = integer()), na_struct(list(x = na_int32()))
  )
  expect_identical(stream$get_next()$length, 0L)
  stream$release()
})

test_that("function_array_stream() validates its validate argument", {
  for (validate in list(NA, logical(), c(TRUE, FALSE), 1L, "TRUE")) {
    expect_error(
      function_array_stream(function() NULL, na_int32(), validate = validate),
      "`validate` must be a single non-NA logical value"
    )
  }
})

test_that("function_array_stream() coerces common array inputs", {
  for (value in list(1:3, as_nanoarrow_array(1:3))) {
    stream <- function_array_stream(function() value, na_int32())
    expect_identical(stream$get_next()$length, 3L)
    stream$release()
  }
})

test_that("function_array_stream() validates schemas and preserves unchecked behavior", {
  malformed <- as_nanoarrow_array(data.frame(x = 1L))
  nanoarrow_array_set_schema(malformed, na_int32(), validate = FALSE)

  stream <- function_array_stream(function() malformed, na_int32())
  expect_error(stream$get_next(), "Expected array with 2 buffer")
  expect_error(stream$get_next(), "Expected array with 2 buffer")
  stream$release()

  stream <- function_array_stream(
    function() malformed, na_int32(), validate = FALSE
  )
  expect_s3_class(stream$get_next(validate = FALSE), "nanoarrow_array")
  stream$release()
})

test_that("function_array_stream() errors become stable stream errors", {
  stream <- function_array_stream(function() stop("callback failed"), na_int32())
  expect_error(stream$get_next(), "callback failed")
  expect_error(stream$get_next(), "callback failed")
  stream$release()
})

test_that("function_array_stream() contains interrupts and re-entrant calls", {
  interrupted <- function() {
    condition <- structure(
      list(message = "test interrupt"),
      class = c("interrupt", "condition")
    )
    signalCondition(condition)
    1L
  }
  stream <- function_array_stream(interrupted, na_int32())
  expect_error(stream$get_next(), "R callback interrupted")
  expect_error(stream$get_next(), "R callback interrupted")
  stream$release()

  stream <- NULL
  stream <- function_array_stream(function() stream$get_next(), na_int32())
  expect_error(stream$get_next(), "Re-entrant call to function array stream")
  expect_error(stream$get_next(), "Re-entrant call to function array stream")
  expect_identical(
    function_array_stream_get_next_on_other_thread(stream), 22L
  )
  stream$release()

  stream <- NULL
  stream <- function_array_stream(
    function() {
      tryCatch(stream$get_next(), error = function(e) NULL)
      1L
    },
    na_int32()
  )
  expect_error(stream$get_next(), "Re-entrant call to function array stream")
  expect_identical(
    function_array_stream_get_next_on_other_thread(stream), 22L
  )
  stream$release()
})

test_that("function_array_stream() contains abort restarts and allows warnings", {
  stream <- function_array_stream(
    function() invokeRestart("abort"), na_int32()
  )
  expect_error(
    stream$get_next(),
    "R callback exited non-locally .*abort restart"
  )
  expect_error(stream$get_next(), "R callback exited non-locally")
  stream$release()

  stream <- function_array_stream(function() {
    warning("callback warning")
    1L
  }, na_int32())
  withr::local_options(warn = 1)
  caller_handled_warning <- FALSE
  warning_output <- capture.output(
    result <- withCallingHandlers(
      stream$get_next(),
      warning = function(w) {
        caller_handled_warning <<- TRUE
        invokeRestart("muffleWarning")
      }
    ),
    type = "message"
  )
  expect_match(paste(warning_output, collapse = "\n"), "callback warning")
  expect_false(caller_handled_warning)
  expect_identical(result$length, 1L)
  stream$release()
})

test_that("function_array_stream() reports conditionMessage failures safely", {
  method <- "conditionMessage.nanoarrow_test_broken"
  had_method <- exists(method, envir = globalenv(), inherits = FALSE)
  if (had_method) {
    old_method <- get(method, envir = globalenv(), inherits = FALSE)
  }
  on.exit(
    if (had_method) {
      assign(method, old_method, envir = globalenv())
    } else {
      rm(list = method, envir = globalenv())
    },
    add = TRUE
  )
  assign(method, function(condition) stop("conditionMessage failed"), globalenv())

  stream <- function_array_stream(
    function() stop(structure(
      list(message = "original error"),
      class = c("nanoarrow_test_broken", "error", "condition")
    )),
    na_int32()
  )
  expect_error(
    stream$get_next(),
    "\\[5\\].*conditionMessage failed"
  )
  expect_error(stream$get_next(), "\\[5\\].*conditionMessage failed")
  stream$release()
})

test_that("function_array_stream() protects callback state across export and GC", {
  value <- 1:3
  stream <- function_array_stream(function() value, na_int32())
  stream_copy <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_export(stream, stream_copy)
  rm(stream)
  gc()
  expect_identical(stream_copy$get_next()$length, 3L)
  stream_copy$release()
})

test_that("function_array_stream() keeps callbacks alive through finalizer export", {
  calls <- 0L
  finalized <- 0L
  stream <- function_array_stream(function() {
    calls <<- calls + 1L
    calls
  }, na_int32())
  stream <- array_stream_set_finalizer(stream, function() {
    finalized <<- finalized + 1L
  })
  stream_copy <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_export(stream, stream_copy)
  rm(stream)
  gc()

  expect_identical(stream_copy$get_next()$length, 1L)
  stream_copy$release()
  stream_copy$release()
  expect_identical(finalized, 1L)
})

test_that("function_array_stream() rejects evaluation on a non-R thread", {
  calls <- 0L
  stream <- function_array_stream(function() {
    calls <<- calls + 1L
    1L
  }, na_int32())
  expect_identical(
    function_array_stream_get_next_on_other_thread(stream),
    5L
  )
  expect_error(stream$get_next(), "Can't evaluate R callback on a non-R thread")
  expect_identical(calls, 0L)
  stream$release()

  stream <- function_array_stream(function() NULL, na_int32())
  expect_null(stream$get_next())
  expect_identical(
    function_array_stream_get_next_on_other_thread(stream), 0L
  )
  stream$release()
})

test_that("function_array_stream() releases preserved state exactly once", {
  gc()
  preserved_empty()
  baseline <- preserved_count()
  stream <- function_array_stream(function() 1L, na_int32())
  after_create <- preserved_count()
  expect_gt(after_create, baseline)
  stream$release()
  expect_lte(preserved_count(), baseline)
  stream$release()
  expect_lte(preserved_count(), baseline)

  stream <- function_array_stream(function() 1L, na_int32())
  rm(stream)
  gc()
  preserved_empty()
  expect_lte(preserved_count(), baseline)
})

test_that("function_array_stream() integrates with stream conversion", {
  stream <- function_array_stream(
    local({
      i <- 0L
      function() {
        i <<- i + 1L
        if (i > 2L) NULL else data.frame(x = i)
      }
    }),
    na_struct(list(x = na_int32()))
  )
  expect_identical(as.data.frame(stream), data.frame(x = 1:2))

  stream <- function_array_stream(
    local({
      i <- 0L
      function() {
        i <<- i + 1L
        if (i > 2L) NULL else data.frame(x = i)
      }
    }),
    na_struct(list(x = na_int32()))
  )
  expect_identical(
    lapply(collect_array_stream(stream), as.data.frame),
    list(data.frame(x = 1L), data.frame(x = 2L))
  )
  stream$release()

  stream <- function_array_stream(
    local({
      i <- 0L
      function() {
        i <<- i + 1L
        if (i > 2L) NULL else data.frame(x = i)
      }
    }),
    na_struct(list(x = na_int32()))
  )
  expect_identical(convert_array_stream(stream), data.frame(x = 1:2))
  stream$release()
})

test_that("function_array_stream() methods are safe after release", {
  stream <- function_array_stream(function() NULL, na_int32())
  stream$release()
  expect_error(stream$get_next())
  expect_error(stream$get_schema())
})

test_that("function_array_stream() reports release from its callback", {
  stream <- NULL
  stream <- function_array_stream(
    function() {
      stream$release()
      1L
    }, na_int32()
  )
  expect_error(stream$get_next(), "\\[22\\].*Array stream has been released")
  expect_false(nanoarrow_pointer_is_valid(stream))
  expect_silent(stream$release())
})

test_that("function_array_stream() defers wrapped release across callback GC", {
  released <- FALSE
  holder <- new.env(parent = emptyenv())
  holder$stream <- NULL
  parent <- function_array_stream(
    function() {
      holder$stream$release()
      gc()
      1L
    },
    na_int32()
  )
  holder$stream <- array_stream_set_finalizer(parent, function() {
    released <<- TRUE
  })

  expect_error(
    holder$stream$get_next(),
    "\\[22\\].*Array stream has been released"
  )
  expect_true(released)
  expect_false(nanoarrow_pointer_is_valid(holder$stream))
})

test_that("function_array_stream() releases preserved state on another thread", {
  gc()
  preserved_empty()
  baseline <- preserved_count()
  stream <- function_array_stream(function() NULL, na_int32())
  expect_gt(preserved_count(), baseline)

  release_array_stream_on_other_thread(stream)
  expect_false(nanoarrow_pointer_is_valid(stream))
  # Both R references must remain queued until the main thread drains them.
  expect_identical(preserved_empty(), 2)
  expect_lte(preserved_count(), baseline)
})

test_that("wrapped stream release is safe during its finalizer", {
  stream <- NULL
  finalizer_results <- new.env(parent = emptyenv())
  stream <- function_array_stream(function() NULL, na_int32())
  stream <- array_stream_set_finalizer(stream, function() {
    finalizer_results$schema <- tryCatch(stream$get_schema(), error = conditionMessage)
    finalizer_results$next_result <- tryCatch(stream$get_next(), error = conditionMessage)
    finalizer_results$release <- tryCatch(stream$release(), error = conditionMessage)
  })

  expect_silent(stream$release())
  expect_match(finalizer_results$schema, "already been released")
  expect_match(finalizer_results$next_result, "already been released")
  expect_null(finalizer_results$release)
  expect_false(nanoarrow_pointer_is_valid(stream))
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

test_that("as_nanoarrow_array_stream() works for nanoarrow_array_stream", {
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

test_that("as_nanoarrow_array_stream() works for nanoarrow_array", {
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
