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

#' Convert an object to a nanoarrow array
#'
#' In nanoarrow an 'array' refers to the `struct ArrowArray` definition
#' in the Arrow C data interface. At the R level, we attach a
#' [schema][as_nanoarrow_schema] such that functionally the nanoarrow_array
#' class can be used in a similar way as an [arrow::Array]. Note that in
#' nanoarrow an [arrow::RecordBatch] and a non-nullable [arrow::StructArray]
#' are represented identically.
#'
#' @param x An object to convert to a array
#' @param schema An optional schema used to enforce conversion to a particular
#'   type. Defaults to [infer_nanoarrow_schema()].
#' @param ... Passed to S3 methods
#'
#' @return An object of class 'nanoarrow_array'
#' @export
#'
#' @examples
#' (array <- as_nanoarrow_array(1:5))
#' as.vector(array)
#'
#' (array <- as_nanoarrow_array(data.frame(x = 1:5)))
#' as.data.frame(array)
#'
as_nanoarrow_array <- function(x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array")
}

#' @export
as.vector.nanoarrow_array <- function(x, mode = "any") {
  stopifnot(identical(mode, "any"))
  convert_array(x, to = infer_nanoarrow_ptype(x))
}

#' @export
as.data.frame.nanoarrow_array <- function(x, ...) {
  schema <- infer_nanoarrow_schema(x)
  if (schema$format != "+s") {
    stop(
      sprintf(
        "Can't convert array with type %s to data.frame()",
        nanoarrow_schema_formatted(schema)
      )
    )
  }

  .Call(nanoarrow_c_convert_array, x, NULL)
}

# exported in zzz.R
as_tibble.nanoarrow_array <- function(x, ...) {
  tibble::as_tibble(as.data.frame.nanoarrow_array(x), ...)
}

#' @export
as_nanoarrow_array.default <- function(x, ..., schema = NULL) {
  # For now, use arrow's conversion for everything
  if (is.null(schema)) {
    as_nanoarrow_array(arrow::as_arrow_array(x))
  } else {
    schema <- as_nanoarrow_schema(schema)
    as_nanoarrow_array(arrow::as_arrow_array(x, type = arrow::as_data_type(schema)))
  }
}

#' @export
infer_nanoarrow_schema.nanoarrow_array <- function(x, ...) {
  .Call(nanoarrow_c_infer_schema_array, x) %||%
    stop("nanoarrow_array() has no associated schema")
}

nanoarrow_array_set_schema <- function(array, schema, validate = TRUE) {
  .Call(nanoarrow_c_array_set_schema, array, schema, as.logical(validate)[1])
  invisible(array)
}

#' @importFrom utils str
#' @export
str.nanoarrow_array <- function(object, ...) {
  cat(sprintf("%s\n", format(object, .recursive = FALSE)))

  if (nanoarrow_pointer_is_valid(object)) {
    # Use the str() of the list version but remove the first
    # line of the output ("List of 6")
    info <- nanoarrow_array_proxy_safe(object)
    raw_str_output <- utils::capture.output(str(info, ...))
    cat(paste0(raw_str_output[-1], collapse = "\n"))
    cat("\n")
  }

  invisible(object)
}

#' @export
print.nanoarrow_array <- function(x, ...) {
  str(x, ...)
  invisible(x)
}

#' @export
format.nanoarrow_array <- function(x, ..., .recursive = TRUE) {
  if (nanoarrow_pointer_is_valid(x)) {
    schema <- .Call(nanoarrow_c_infer_schema_array, x)
    if (is.null(schema)) {
      sprintf("<nanoarrow_array <unknown schema>[%s]>", x$length)
    } else {
      sprintf(
        "<nanoarrow_array %s[%s]>",
        nanoarrow_schema_formatted(schema, .recursive),
        x$length
      )
    }
  } else {
    "<nanoarrow_array[invalid pointer]>"
  }
}


# This is the list()-like interface to nanoarrow_array that allows $ and [[
# to make nice auto-complete for the array fields


#' @export
length.nanoarrow_array <- function(x, ...) {
  6L
}

#' @export
names.nanoarrow_array <- function(x, ...) {
  c("length",  "null_count", "offset", "buffers", "children", "dictionary")
}

#' @export
`[[.nanoarrow_array` <- function(x, i, ...) {
  nanoarrow_array_proxy_safe(x)[[i]]
}

#' @export
`$.nanoarrow_array` <- function(x, i, ...) {
  nanoarrow_array_proxy_safe(x)[[i]]
}

# A version of nanoarrow_array_proxy() that is less likely to error for invalid
# arrays and/or schemas
nanoarrow_array_proxy_safe <- function(array, recursive = FALSE) {
  schema <- .Call(nanoarrow_c_infer_schema_array, array)
  tryCatch(
    nanoarrow_array_proxy(array, schema = schema, recursive = recursive),
    error = function(...) nanoarrow_array_proxy(array, recursive = recursive)
  )
}

nanoarrow_array_proxy <- function(array, schema = NULL, recursive = FALSE) {
  if (!is.null(schema)) {
    array_view <- .Call(nanoarrow_c_array_view, array, schema)
    result <- .Call(nanoarrow_c_array_proxy, array, array_view, recursive)

    # Pass on some information from the schema if we have it
    if (!is.null(result$dictionary)) {
      nanoarrow_array_set_schema(result$dictionary, schema$dictionary)
    }

    names(result$children) <- names(schema$children)

    if (!recursive) {
      result$children <- Map(
        nanoarrow_array_set_schema,
        result$children,
        schema$children
      )
    }
  } else {
    result <- .Call(nanoarrow_c_array_proxy, array, NULL, recursive)
  }

  # Recursive-ness of the dictionary is handled here because it's not
  # part of the array view
  if (recursive && !is.null(result$dictionary)) {
    result$dictionary <- nanoarrow_array_proxy(
      result$dictionary,
      schema = schema$dictionary,
      recursive = TRUE
    )
  }

  result
}
