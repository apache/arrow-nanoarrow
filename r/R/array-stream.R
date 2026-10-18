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

#' Create ArrayStreams from batches
#'
#' @param batches A [list()] of [nanoarrow_array][as_nanoarrow_array] objects
#'   or objects that can be coerced via [as_nanoarrow_array()].
#' @param schema A [nanoarrow_schema][as_nanoarrow_schema] or `NULL` to guess
#'   based on the first schema.
#' @param validate Use `FALSE` to skip the validation step (i.e., if you
#'   know that the arrays are valid).
#'
#' @return An [nanoarrow_array_stream][as_nanoarrow_array_stream]
#' @export
#'
#' @examples
#' (stream <- basic_array_stream(list(data.frame(a = 1, b = 2))))
#' as.data.frame(stream$get_next())
#' stream$get_next()
#'
basic_array_stream <- function(batches, schema = NULL, validate = TRUE) {
  # Error for everything except a bare list (e.g., so that calling with
  # a data.frame() does not unintentionally loop over columns)
  if (!identical(class(batches), "list")) {
    stop("`batches` must be an unclassed `list()`")
  }

  batches <- lapply(batches, as_nanoarrow_array, schema = schema)

  if (is.null(schema) && length(batches) > 0) {
    schema <- infer_nanoarrow_schema(batches[[1]])
  } else if (is.null(schema)) {
    stop("Can't infer schema from first batch if there are zero batches")
  }

  .Call(nanoarrow_c_basic_array_stream, batches, schema, validate)
}

#' Create an array stream from an R function
#'
#' The function is called once for each requested array. It should return an
#' object coercible by [as_nanoarrow_array()], or `NULL` to indicate end of
#' stream. The schema is required because it is available before the first
#' array is requested. Neither construction nor schema access calls `fun`.
#' After `NULL` or an error is returned, the callback is not called again.
#'
#' Evaluation is isolated in a top-level R context so non-local exits (for
#' example, invoking an `abort` restart) cannot escape through the C stream
#' interface. Calling handlers in the caller are therefore not applied to
#' callback evaluation. The callback is evaluated only on R's main thread.
#' Warnings are handled in this isolated context, unless handled within `fun`.
#' The stream retains `fun` until it is released; explicitly release streams
#' when `fun` or its environment refers to the stream itself.
#'
#' @param fun A function called with no arguments to produce each array.
#' @param schema A schema coercible by [as_nanoarrow_schema()], used to coerce
#'   and validate each result.
#' @param validate A single `TRUE` or `FALSE`. Use `FALSE` to skip validation
#'   within the callback. The stream's `get_next()` method can validate the
#'   returned array independently.
#'
#' @return An [nanoarrow_array_stream][as_nanoarrow_array_stream]
#' @export
#'
#' @examples
#' i <- 0L
#' stream <- function_array_stream(
#'   function() {
#'     i <<- i + 1L
#'     if (i > 2L) NULL else mtcars[i, , drop = FALSE]
#'   },
#'   infer_nanoarrow_schema(mtcars)
#' )
#' as.data.frame(stream$get_next())
#' as.data.frame(stream$get_next())
#' stream$get_next()
#' stream$release()
#'
function_array_stream <- function(fun, schema, validate = TRUE) {
  stopifnot(is.function(fun))
  schema <- as_nanoarrow_schema(schema)
  if (!is.logical(validate) || length(validate) != 1L || is.na(validate)) {
    stop("`validate` must be a single non-NA logical value")
  }

  .Call(nanoarrow_c_function_array_stream, fun, schema, validate)
}

function_array_stream_get_next_wrapper <- function(fun, schema, validate) {
  result <- fun()
  if (is.null(result)) {
    NULL
  } else {
    array <- as_nanoarrow_array(result, schema = schema)
    if (validate) {
      nanoarrow_array_set_schema(array, schema, validate = TRUE)
    }
    array
  }
}

function_array_stream_get_next_on_other_thread <- function(x) {
  .Call(nanoarrow_c_function_array_stream_get_next_on_other_thread, x)
}

#' Register an array stream finalizer
#'
#' In some cases, R functions that return a [nanoarrow_array_stream][as_nanoarrow_array_stream]
#' may require that the scope of some other object outlive that of the array
#' stream. If there is a need for that object to be released deterministically
#' (e.g., to close open files), you can register a function to run after the
#' stream's release callback is invoked from the R thread. Note that this
#' finalizer will **not** be run if the stream's release callback is invoked
#' from a **non**-R thread. In this case, the finalizer and its chain of
#' environments will be garbage-collected when `nanoarrow:::preserved_empty()`
#' is run.
#'
#' @param array_stream A [nanoarrow_array_stream][as_nanoarrow_array_stream]
#' @param finalizer A function that will be called with zero arguments.
#'
#' @return A newly allocated `array_stream` whose release callback will call
#'   the supplied finalizer.
#' @export
#'
#' @examples
#' stream <- array_stream_set_finalizer(
#'   basic_array_stream(list(1:5)),
#'   function() message("All done!")
#' )
#' stream$release()
#'
array_stream_set_finalizer <- function(array_stream, finalizer) {
  stopifnot(is.function(finalizer))

  prot <- new.env(parent = emptyenv())
  prot$array_stream_finalizer <- finalizer
  class(prot) <- "nanoarrow_array_stream_finalizer"

  nanoarrow_pointer_set_protected(array_stream, prot)
  out <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_export(array_stream, out)
  out
}

#' Convert an object to a nanoarrow array_stream
#'
#' In nanoarrow, an 'array stream' corresponds to the `struct ArrowArrayStream`
#' as defined in the Arrow C Stream interface. This object is used to represent
#' a stream of [arrays][as_nanoarrow_array] with a common
#' [schema][as_nanoarrow_schema]. This is similar to an
#' [arrow::RecordBatchReader] except it can be used to represent a stream of
#' any type (not just record batches). Note that a stream of record batches
#' and a stream of non-nullable struct arrays are represented identically.
#' Also note that array streams are mutable objects and are passed by
#' reference and not by value.
#'
#' @param x An object to convert to a array_stream
#' @param ... Passed to S3 methods
#' @inheritParams as_nanoarrow_array
#'
#' @return An object of class 'nanoarrow_array_stream'
#' @export
#'
#' @examples
#' (stream <- as_nanoarrow_array_stream(data.frame(x = 1:5)))
#' stream$get_schema()
#' stream$get_next()
#'
#' # The last batch is returned as NULL
#' stream$get_next()
#'
#' # Release the stream
#' stream$release()
#'
as_nanoarrow_array_stream <- function(x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array_stream")
}

#' @export
as_nanoarrow_array_stream.nanoarrow_array_stream <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    return(x)
  }

  inferred_schema <- infer_nanoarrow_schema(x)
  if (nanoarrow_schema_identical(schema, inferred_schema)) {
    return(x)
  }

  NextMethod()
}

#' @export
as_nanoarrow_array_stream.nanoarrow_array <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    return(basic_array_stream(list(x), validate = FALSE))
  }

  inferred_schema <- infer_nanoarrow_schema(x)
  if (nanoarrow_schema_identical(schema, inferred_schema)) {
    return(basic_array_stream(list(x), validate = FALSE))
  }

  as_nanoarrow_array_stream(
    as_nanoarrow_array_stream(x),
    schema = schema
  )
}

#' @export
as_nanoarrow_array_stream.default <- function(x, ..., schema = NULL) {
  assert_arrow_installed("default coerce to nanoarrow_array_stream")

  as_nanoarrow_array_stream(
    arrow::as_record_batch_reader(x, ..., schema = arrow::as_schema(schema)),
    schema = schema
  )
}

#' @export
as_nanoarrow_array_stream.data.frame <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  } else {
    schema <- as_nanoarrow_schema(schema)
  }

  x <- as_nanoarrow_array(x, schema = schema)
  basic_array_stream(list(x), schema = schema)
}

#' @export
infer_nanoarrow_schema.nanoarrow_array_stream <- function(x, ...) {
  x$get_schema()
}

#' @export
as.data.frame.nanoarrow_array_stream <- function(x, ...) {
  # Always release the input: we are always consuming the entire stream.
  # For more fine-grained behaviour on error, one can use
  # convert_array_stream()
  on.exit(x$release())

  to <- infer_nanoarrow_ptype(x$get_schema())
  if (!inherits(to, "data.frame")) {
    stop("Can't convert non-struct array stream to data.frame")
  }

  convert_array_stream(x, to)
}

#' @export
as.vector.nanoarrow_array_stream <- function(x, mode) {
  on.exit(x$release())
  convert_array_stream(x)
}

#' @importFrom utils str
#' @export
str.nanoarrow_array_stream <- function(object, ...) {
  cat(sprintf("%s\n", format(object)))

  if (nanoarrow_pointer_is_valid(object)) {
    # Use the str() of the list version but remove the first
    # line of the output ("List of 2")
    info <- list(
      get_schema = object$get_schema,
      get_next = object$get_next,
      release = object$release
    )
    raw_str_output <- utils::capture.output(str(info, ..., give.attr = FALSE))
    cat(paste0(raw_str_output[-1], collapse = "\n"))
    cat("\n")
  }

  invisible(object)
}

#' @export
print.nanoarrow_array_stream <- function(x, ...) {
  str(x, ...)
  invisible(x)
}

#' @export
format.nanoarrow_array_stream <- function(x, ...) {
  if (nanoarrow_pointer_is_valid(x)) {
    tryCatch(
      sprintf("<nanoarrow_array_stream %s>", nanoarrow_schema_formatted(x$get_schema())),
      error = function(...) "<nanoarrow_array_stream[<error calling get_schema()]>"
    )

  } else {
    "<nanoarrow_array_stream[invalid pointer]>"
  }
}

# This is the list()-like interface to nanoarrow_array_stream that allows $ and [[
# to make nice auto-complete when interacting in an IDE

#' @export
length.nanoarrow_array_stream <- function(x, ...) {
  3L
}

#' @export
names.nanoarrow_array_stream <- function(x, ...) {
  c("get_schema", "get_next", "release")
}

#' @export
`[[.nanoarrow_array_stream` <- function(x, i, ...) {
  force(x)
  if (identical(i, "get_schema") || isTRUE(i == 1L)) {
    function() .Call(nanoarrow_c_array_stream_get_schema, x)
  } else if (identical(i, "get_next") || isTRUE(i == 2L)) {
    function(schema = x$get_schema(), validate = TRUE) {
      array <- .Call(nanoarrow_c_array_stream_get_next, x)
      if (!nanoarrow_pointer_is_valid(array)) {
        return(NULL)
      }

      nanoarrow_array_set_schema(array, schema, validate = validate)
      array
    }
  } else if (identical(i, "release") || isTRUE(i == 3L)) {
    function() nanoarrow_pointer_release(x)
  } else {
    NULL
  }
}

#' @export
`$.nanoarrow_array_stream` <- function(x, i, ...) {
  x[[i]]
}
