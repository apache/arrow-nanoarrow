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

#' Convert an object to a nanoarrow schema
#'
#' In nanoarrow a 'schema' refers to a `struct ArrowSchema` as defined in the
#' Arrow C Data interface. This data structure can be used to represent an
#' [arrow::schema()], an [arrow::field()], or an `arrow::DataType`. Note that
#' in nanoarrow, an [arrow::schema()] and a non-nullable [arrow::struct()]
#' are represented identically.
#'
#' @param x An object to convert to a schema
#' @param recursive Use `TRUE` to include a `children` member when parsing
#'   schemas.
#' @param ... Passed to S3 methods
#'
#' @return An object of class 'nanoarrow_schema'
#' @export
#'
#' @examples
#' infer_nanoarrow_schema(integer())
#' infer_nanoarrow_schema(data.frame(x = integer()))
#'
as_nanoarrow_schema <- function(x, ...) {
  UseMethod("as_nanoarrow_schema")
}

#' @export
as_nanoarrow_schema.nanoarrow_schema <- function(x, ...) {
  x
}

#' @rdname as_nanoarrow_schema
#' @export
infer_nanoarrow_schema <- function(x, ...) {
  UseMethod("infer_nanoarrow_schema")
}

#' @export
infer_nanoarrow_schema.default <- function(x, ...) {
  as_nanoarrow_schema(arrow::infer_type(x, ...))
}

#' @rdname as_nanoarrow_schema
#' @export
nanoarrow_schema_parse <- function(x, recursive = FALSE) {
  parsed <- .Call(nanoarrow_c_schema_parse, as_nanoarrow_schema(x))
  parsed_null <- vapply(parsed, is.null, logical(1))
  result <- parsed[!parsed_null]

  if (recursive && !is.null(x$children)) {
    result$children <- lapply(x$children, nanoarrow_schema_parse, TRUE)
  }

  result
}

#' @importFrom utils str
#' @export
str.nanoarrow_schema <- function(object, ...) {
  cat(sprintf("%s\n", format(object, .recursive = FALSE)))

  if (nanoarrow_pointer_is_valid(object)) {
    # Use the str() of the list version but remove the first
    # line of the output ("List of 6")
    info <- nanoarrow_schema_proxy(object)
    raw_str_output <- utils::capture.output(str(info, ...))
    cat(paste0(raw_str_output[-1], collapse = "\n"))
    cat("\n")
  }

  invisible(object)
}

#' @export
print.nanoarrow_schema <- function(x, ...) {
  str(x, ...)
  invisible(x)
}

#' @export
format.nanoarrow_schema <- function(x, ..., .recursive = TRUE) {
  sprintf(
    "<nanoarrow_schema %s>",
    nanoarrow_schema_formatted(x, .recursive)
  )
}

# This is the list()-like interface to nanoarrow_schema that allows $ and [[
# to make nice auto-complete for the schema fields

#' @export
length.nanoarrow_schema <- function(x, ...) {
  6L
}

#' @export
names.nanoarrow_schema <- function(x, ...) {
  c("format", "name", "metadata", "flags", "children", "dictionary")
}

#' @export
`[[.nanoarrow_schema` <- function(x, i, ...) {
  nanoarrow_schema_proxy(x)[[i]]
}

#' @export
`$.nanoarrow_schema` <- function(x, i, ...) {
  nanoarrow_schema_proxy(x)[[i]]
}

nanoarrow_schema_formatted <- function(x, recursive = TRUE) {
  .Call(nanoarrow_c_schema_format, x, as.logical(recursive)[1])
}

nanoarrow_schema_proxy <- function(schema, recursive = FALSE) {
  result <- .Call(nanoarrow_c_schema_to_list, schema)
  if (recursive && !is.null(schema$children)) {
    result$children <- lapply(
      schema$children,
      nanoarrow_schema_proxy,
      recursive = TRUE
    )
  }

  if (recursive && !is.null(schema$dictionary)) {
    result$dictionary <- nanoarrow_schema_proxy(schema$dictionary, recursive = TRUE)
  }

  result$metadata <- list_of_raw_to_metadata(result$metadata)

  result
}

list_of_raw_to_metadata <- function(metadata) {
  lapply(metadata, function(x) {
    if (is.character(x) || any(x == 0)) {
      x
    } else {
      x_str <- iconv(list(x), from = "UTF-8", to = "UTF-8", mark = TRUE)[[1]]
      if (is.na(x_str)) x else x_str
    }
  })
}
