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
    x
  } else {
    NextMethod()
  }
}

#' @export
as_nanoarrow_array_stream.default <- function(x, ..., schema = NULL) {
  as_nanoarrow_array_stream(
    arrow::as_record_batch_reader(x, ...),
    schema = schema
  )
}

#' @export
infer_nanoarrow_schema.nanoarrow_array_stream <- function(x, ...) {
  x$get_schema()
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
      sprintf("<nanoarrow_array_stream[%s]>", x$get_schema()$format),
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
