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
#' class can be used in a similar way as an `arrow::Array`. Note that in
#' nanoarrow an `arrow::RecordBatch` and a non-nullable `arrow::StructArray`
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

# See as-array.R for S3 method implementations

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
infer_nanoarrow_schema.nanoarrow_array <- function(x, ...) {
  .Call(nanoarrow_c_infer_schema_array, x) %||%
    stop("nanoarrow_array() has no associated schema")
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

#' @export
`[[<-.nanoarrow_array` <- function(x, i, value) {
  if (is.numeric(i) && isTRUE(i %in% 1:6)) {
    i <- names.nanoarrow_array()[[i]]
  }

  if (is.character(i) && (length(i) == 1L) && !is.na(i)) {
    new_values <- list(value)
    names(new_values) <- i
    return(nanoarrow_array_modify(x, new_values))
  }

  stop("`i` must be character(1) or integer(1) %in% 1:6")
}

#' @export
`$<-.nanoarrow_array` <- function(x, i, value) {
  new_values <- list(value)
  names(new_values) <- i
  nanoarrow_array_modify(x, new_values)
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

    names(result$children) <- names(schema$children)

    if (!recursive) {
      # Pass on some information from the schema if we have it
      result$children <- Map(
        nanoarrow_array_set_schema,
        result$children,
        schema$children
      )

      if (!is.null(result$dictionary)) {
        nanoarrow_array_set_schema(result$dictionary, schema$dictionary)
      }
    }
  } else {
    result <- .Call(nanoarrow_c_array_proxy, array, NULL, recursive)
  }

  result
}


#' Modify nanoarrow arrays
#'
#' Create a new array or from an existing array, modify one or more parameters.
#' When importing an array from elsewhere, `nanoarrow_array_set_schema()` is
#' useful to attach the data type information to the array (without this
#' information there is little that nanoarrow can do with the array since its
#' content cannot be otherwise interpreted). `nanoarrow_array_modify()` can
#' create a shallow copy and modify various parameters to create a new array,
#' including setting children and buffers recursively. These functions power the
#' `$<-` operator, which can modify one parameter at a time.
#'
#' @param array A [nanoarrow_array][as_nanoarrow_array].
#' @param schema A [nanoarrow_schema][as_nanoarrow_schema] to attach to this
#'   `array`.
#' @param new_values A named `list()` of values to replace.
#' @param validate Use `FALSE` to skip validation. Skipping validation may
#'   result in creating an array that will crash R.
#'
#' @return
#'   - `nanoarrow_array_init()` returns a possibly invalid but initialized
#'     array with a given `schema`.
#'   - `nanoarrow_array_set_schema()` returns `array`, invisibly. Note that
#'      `array` is modified in place by reference.
#'   - `nanoarrow_array_modify()` returns a shallow copy of `array` with the
#' modified parameters such that the original array remains valid.
#' @export
#'
#' @examples
#' nanoarrow_array_init(na_string())
#'
#' # Modify an array using $ and <-
#' array <- as_nanoarrow_array(1:5)
#' array$length <- 4
#' as.vector(array)
#'
#' # Modify potentially more than one component at a time
#' array <- as_nanoarrow_array(1:5)
#' as.vector(nanoarrow_array_modify(array, list(length = 4)))
#'
#' # Attach a schema to an array
#' array <- as_nanoarrow_array(-1L)
#' nanoarrow_array_set_schema(array, na_uint32())
#' as.vector(array)
#'
nanoarrow_array_init <- function(schema) {
  .Call(nanoarrow_c_array_init, schema)
}

#' @rdname nanoarrow_array_init
#' @export
nanoarrow_array_set_schema <- function(array, schema, validate = TRUE) {
  .Call(nanoarrow_c_array_set_schema, array, schema, as.logical(validate)[1])
  invisible(array)
}

#' @rdname nanoarrow_array_init
#' @export
nanoarrow_array_modify <- function(array, new_values, validate = TRUE) {
  array <- as_nanoarrow_array(array)

  if (length(new_values) == 0) {
    return(array)
  }

  # Make sure new_values has names to iterate over
  new_names <- names(new_values)
  if (is.null(new_names) || all(new_names == "", na.rm = TRUE)) {
    stop("`new_values` must be named")
  }

  # Make a copy and modify it. This is a deep copy in the sense that all
  # children are modifiable; however, it's a shallow copy in the sense that
  # none of the buffers are copied.
  schema <- .Call(nanoarrow_c_infer_schema_array, array)
  array_copy <- array_shallow_copy(array, schema, validate = validate)

  for (i in seq_along(new_values)) {
    nm <- new_names[i]
    value <- new_values[[i]]

    switch(
      nm,
      length = .Call(nanoarrow_c_array_set_length, array_copy, as.double(value)),
      null_count = .Call(nanoarrow_c_array_set_null_count, array_copy, as.double(value)),
      offset = .Call(nanoarrow_c_array_set_offset, array_copy, as.double(value)),
      buffers = {
        value <- lapply(value, as_nanoarrow_buffer)
        .Call(nanoarrow_c_array_set_buffers, array_copy, value)
      },
      children = {
        value <- lapply(value, as_nanoarrow_array)
        value_copy <- lapply(value, array_shallow_copy, validate = validate)
        .Call(nanoarrow_c_array_set_children, array_copy, value_copy)

        if (!is.null(schema)) {
          schema <- nanoarrow_schema_modify(
            schema,
            list(children = lapply(value, infer_nanoarrow_schema)),
            validate = validate
          )
        }
      },
      dictionary = {
        if (!is.null(value)) {
          value <- as_nanoarrow_array(value)
          value_copy <- array_shallow_copy(value, validate = validate)
        } else {
          value_copy <- NULL
        }

        .Call(nanoarrow_c_array_set_dictionary, array_copy, value_copy)

        if (!is.null(schema) && !is.null(value)) {
          schema <- nanoarrow_schema_modify(
            schema,
            list(dictionary = infer_nanoarrow_schema(value)),
            validate = validate
          )
        } else if (!is.null(schema)) {
          schema <- nanoarrow_schema_modify(
            schema,
            list(dictionary = NULL),
            validate = validate
          )
        }
      },
      stop(sprintf("Can't modify array[[%s]]: does not exist", deparse(nm)))
    )
  }

  # Finish building (e.g., ensure pointers are flushed)
  .Call(nanoarrow_c_array_finish_building, array_copy)

  # Validate if requested
  if (!is.null(schema) && validate) {
    array_copy <- .Call(nanoarrow_c_array_validate_after_modify, array_copy, schema)
  }

  if (!is.null(schema)) {
    nanoarrow_array_set_schema(array_copy, schema, validate = validate)
  }

  array_copy
}

array_shallow_copy <- function(array, schema = NULL, validate = TRUE) {
  array_copy <- nanoarrow_allocate_array()
  nanoarrow_pointer_export(array, array_copy)
  schema <- schema %||% .Call(nanoarrow_c_infer_schema_array, array)

  # For validation, use some of the infrastructure we already have in place
  # to make sure array_copy knows how long each buffer is
  if (!is.null(schema) && validate) {
    copy_buffers_recursive(array, array_copy)
  }

  array_copy
}

copy_buffers_recursive <- function(array, array_copy) {
  proxy <- nanoarrow_array_proxy_safe(array)
  proxy_copy <- nanoarrow_array_proxy(array_copy)

  .Call(nanoarrow_c_array_set_buffers, array_copy, proxy$buffers)

  for (i in seq_along(proxy$children)) {
    copy_buffers_recursive(proxy$children[[i]], proxy_copy$children[[i]])
  }

  if (!is.null(proxy$dictionary)) {
    copy_buffers_recursive(proxy$dictionary, proxy_copy$dictionary)
  }
}
