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
#' @param new_values New schema component to assign
#' @param validate Use `FALSE` to skip schema validation
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
  cls <- paste(class(x), collapse = "/")
  stop(sprintf("Can't infer Arrow type for object of class %s", cls))
}

#' @export
infer_nanoarrow_schema.raw <- function(x, ...) {
  na_uint8()
}

#' @export
infer_nanoarrow_schema.logical <- function(x, ...) {
  na_bool()
}

#' @export
infer_nanoarrow_schema.integer <- function(x, ...) {
  na_int32()
}

#' @export
infer_nanoarrow_schema.double <- function(x, ...) {
  na_double()
}

#' @export
infer_nanoarrow_schema.character <- function(x, ...) {
  if (length(x) > 0 && sum(nchar(x, type = "bytes"), na.rm = TRUE) > .Machine$integer.max) {
    na_large_string()
  } else {
    na_string()
  }
}

#' @export
infer_nanoarrow_schema.factor <- function(x, ...) {
  na_dictionary(
    infer_nanoarrow_schema(levels(x)),
    na_int32(),
    ordered = is.ordered(x)
  )
}

#' @export
infer_nanoarrow_schema.POSIXct <- function(x, ...) {
  tz <- attr(x, "tzone")
  if (is.null(tz) || identical(tz, "")) {
    tz <- Sys.timezone()
  }

  na_timestamp(timezone = tz)
}

#' @export
infer_nanoarrow_schema.POSIXlt <- function(x, ...) {
  infer_nanoarrow_schema(new_data_frame(x, length(x)))
}

#' @export
infer_nanoarrow_schema.Date <- function(x, ...) {
  na_date32()
}

#' @export
infer_nanoarrow_schema.difftime <- function(x, ...) {
  # A balance between safety for large time ranges (not overflowing)
  # and safety for small time ranges (not truncating)
  na_duration(unit = "us")
}

#' @export
infer_nanoarrow_schema.data.frame <- function(x, ...) {
  na_struct(lapply(x, infer_nanoarrow_schema), nullable = FALSE)
}

#' @export
infer_nanoarrow_schema.hms <- function(x, ...) {
  # As a default, ms is safer than s and less likely to truncate
  na_time32(unit = "ms")
}

#' @export
infer_nanoarrow_schema.blob <- function(x, ...) {
  if (length(x) > 0 && sum(lengths(x)) > .Machine$integer.max) {
    na_large_binary()
  } else {
    na_binary()
  }
}

#' @export
infer_nanoarrow_schema.vctrs_unspecified <- function(x, ...) {
  na_na()
}

#' @export
infer_nanoarrow_schema.vctrs_list_of <- function(x, ...) {
  child_type <- infer_nanoarrow_schema(attr(x, "ptype"))
  if (length(x) > 0 && sum(lengths(x)) > .Machine$integer.max) {
    na_large_list(child_type)
  } else {
    na_list(child_type)
  }
}

#' @rdname as_nanoarrow_schema
#' @export
nanoarrow_schema_parse <- function(x, recursive = FALSE) {
  parsed <- .Call(nanoarrow_c_schema_parse, as_nanoarrow_schema(x))
  parsed_null <- vapply(parsed, is.null, logical(1))
  result <- parsed[!parsed_null]

  if (recursive && length(x$children) > 0) {
    result$children <- lapply(x$children, nanoarrow_schema_parse, TRUE)
  }

  result
}

#' @rdname as_nanoarrow_schema
#' @export
nanoarrow_schema_modify <- function(x, new_values, validate = TRUE) {
  schema <- as_nanoarrow_schema(x)

  if (length(new_values) == 0) {
    return(schema)
  }

  # Make sure new_values has names to iterate over
  new_names <- names(new_values)
  if (is.null(new_names) || all(new_names == "", na.rm = TRUE)) {
    stop("`new_values` must be named")
  }

  # Make a deep copy and modify it. Possibly not as efficient as it could be
  # but it's unclear to what degree performance is an issue for R-level
  # schema modification.
  schema_deep_copy <- nanoarrow_allocate_schema()
  nanoarrow_pointer_export(schema, schema_deep_copy)

  for (i in seq_along(new_values)) {
    nm <- new_names[i]
    value <- new_values[[i]]

    switch(
      nm,
      format = .Call(
        nanoarrow_c_schema_set_format,
        schema_deep_copy,
        as.character(value)
      ),
      name = {
        if (!is.null(value)) {
          value <- as.character(value)
        }

        .Call(nanoarrow_c_schema_set_name, schema_deep_copy, value)
      },
      flags = .Call(
        nanoarrow_c_schema_set_flags,
        schema_deep_copy,
        as.integer(value)
      ),
      metadata = .Call(
        nanoarrow_c_schema_set_metadata,
        schema_deep_copy,
        as.list(value)
      ),
      children = {
        if (!is.null(value)) {
          value <- lapply(value, as_nanoarrow_schema)
        }

        .Call(nanoarrow_c_schema_set_children, schema_deep_copy, value)
      },
      dictionary = {
        if (!is.null(value)) {
          value <- as_nanoarrow_schema(value)
        }

        .Call(nanoarrow_c_schema_set_dictionary, schema_deep_copy, value)
      },
      stop(sprintf("Can't modify schema[[%s]]: does not exist", deparse(nm)))
    )
  }

  if (validate) {
    nanoarrow_schema_parse(schema_deep_copy, recursive = FALSE)
  }

  schema_deep_copy
}

nanoarrow_schema_identical <- function(x, y) {
  identical(x, y) ||
    identical(
      nanoarrow_schema_proxy(x, recursive = TRUE),
      nanoarrow_schema_proxy(y, recursive = TRUE)
    )
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

#' @export
`[[<-.nanoarrow_schema` <- function(x, i, value) {
  if (is.numeric(i) && isTRUE(i %in% 1:6)) {
    i <- names.nanoarrow_schema()[[i]]
  }

  if (is.character(i) && (length(i) == 1L) && !is.na(i)) {
    new_values <- list(value)
    names(new_values) <- i
    return(nanoarrow_schema_modify(x, new_values))
  }

  stop("`i` must be character(1) or integer(1) %in% 1:6")
}

#' @export
`$<-.nanoarrow_schema` <- function(x, i, value) {
  new_values <- list(value)
  names(new_values) <- i
  nanoarrow_schema_modify(x, new_values)
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
