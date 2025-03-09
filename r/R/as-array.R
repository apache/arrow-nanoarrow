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

#' @export
as_nanoarrow_array.default <- function(x, ..., schema = NULL, .from_c = FALSE) {
  # If we're coming from C it's because we've tried all the internal conversions
  # and no suitable S3 method was found or the x--schema combination is not
  # implemented in nanoarrow. Try arrow::as_arrow_array().
  if (.from_c) {
    # Give extension types a chance to handle conversion
    parsed <- .Call(nanoarrow_c_schema_parse, schema)

    if (!is.null(parsed$extension_name)) {
      spec <- resolve_nanoarrow_extension(parsed$extension_name)
      return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
    }

    assert_arrow_installed(
      sprintf(
        "create %s array from object of type %s",
        nanoarrow_schema_formatted(schema),
        paste0(class(x), collapse = "/")
      )
    )

    result <- as_nanoarrow_array(
      arrow::as_arrow_array(
        x,
        type = arrow::as_data_type(schema)
      )
    )

    # Skip nanoarrow_pointer_export() for these arrays since we know there
    # are no external references to them
    class(result) <- c("nanoarrow_array_dont_export", class(result))

    return(result)
  }

  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  } else {
    schema <- as_nanoarrow_schema(schema)
  }

  .Call(nanoarrow_c_as_array_default, x, schema)
}

#' @export
as_nanoarrow_array.nanoarrow_array <- function(x, ..., schema = NULL) {
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
as_nanoarrow_array.integer64 <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  switch(
    parsed$type,
    int64 = ,
    uint64 = {
      if (anyNA(x)) {
        is_valid_lgl <- is.finite(x)
        is_valid <- as_nanoarrow_array(is_valid_lgl, schema = na_bool())$buffers[[2]]
        na_count <- length(x) - sum(is_valid_lgl)
      } else {
        is_valid <- NULL
        na_count <- 0
      }

      array <- nanoarrow_array_init(schema)
      nanoarrow_array_modify(
        array,
        list(
          length = length(x),
          null_count = na_count,
          buffers = list(is_valid, x)
        )
      )
    },
    as_nanoarrow_array(as.double(x), schema = schema)
  )
}

#' @export
as_nanoarrow_array.POSIXct <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  switch(
    parsed$type,
    timestamp = ,
    duration = {
      multipliers <- c(s = 1.0, ms = 1e3, us = 1e6, ns = 1e9)
      multiplier <- unname(multipliers[parsed$time_unit])
      array <- as_nanoarrow_array(
        as.numeric(x) * multiplier,
        schema = na_type(parsed$storage_type)
      )
      nanoarrow_array_set_schema(array, schema)
      array
    },
    NextMethod()
  )
}

#' @export
as_nanoarrow_array.difftime <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  src_unit <- attr(x, "units")
  switch(
    parsed$type,
    time32 = ,
    time64 = ,
    duration = {
      multipliers <- c(s = 1.0, ms = 1e3, us = 1e6, ns = 1e9)
      src_multipliers <- c(
        secs = 1.0,
        mins = 60.0,
        hours = 3600.0,
        days = 86400.0,
        weeks = 604800.0
      )

      multiplier <- unname(multipliers[parsed$time_unit]) *
        unname(src_multipliers[src_unit])
      array <- as_nanoarrow_array(
        as.numeric(x) * multiplier,
        schema = na_type(parsed$storage_type)
      )
      nanoarrow_array_set_schema(array, schema)
      array
    },
    NextMethod()
  )
}

#' @export
as_nanoarrow_array.blob <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  as_nanoarrow_array(unclass(x), schema = schema)
}

#' @export
as_nanoarrow_array.matrix <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  } else {
    schema <- as_nanoarrow_schema(schema)
  }

  expected_format <- paste0("+w:", ncol(x))
  if (expected_format != schema$format) {
    stop(
      sprintf(
        "Expected schema for matrix with fixed-size list of %d elements but got %s",
        ncol(x),
        nanoarrow_schema_formatted(schema)
      )
    )
  }

  # Raw unclass() doesn't work for matrix()
  row_major_data <- t(x)
  attributes(row_major_data) <- NULL

  child_array <- as_nanoarrow_array(row_major_data, schema = schema$children[[1]])
  array <- nanoarrow_array_init(schema)
  nanoarrow_array_modify(
    array,
    list(
      length = nrow(x),
      null_count = 0,
      buffers = list(NULL),
      children = list(child_array)
    )
  )
}

#' @export
as_nanoarrow_array.data.frame <- function(x, ..., schema = NULL) {
  # We need to override this to prevent the list implementation from handling it
  as_nanoarrow_array.default(x, ..., schema = schema)
}

#' @export
as_nanoarrow_array.list <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name) || parsed$type != "list") {
    return(NextMethod())
  }

  # This R implementation can't handle complex nesting
  if (startsWith(schema$children[[1]]$format, "+")) {
    return(NextMethod())
  }

  array <- nanoarrow_array_init(schema)

  child <- unlist(x, recursive = FALSE, use.names = FALSE)
  if (is.null(child)) {
    child_array <- as_nanoarrow_array.vctrs_unspecified(logical(), schema = na_na())
  } else {
    child_array <- as_nanoarrow_array(child, schema = schema$children[[1]])
  }

  offsets <- c(0L, cumsum(lengths(x)))
  is_na <- vapply(x, is.null, logical(1))
  validity <- as_nanoarrow_array(!is_na)$buffers[[2]]

  nanoarrow_array_modify(
    array,
    list(
      length = length(x),
      null_count = sum(is_na),
      buffers = list(
        validity,
        offsets
      ),
      children = list(
        child_array
      )
    )
  )
}

#' @export
as_nanoarrow_array.Date <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  switch(
    parsed$type,
    date32 = {
      int_vec <- if (is.integer(x)) {
        unclass(x)
      } else {
        as.integer(floor(as.numeric(x)))
      }

      storage <- as_nanoarrow_array(
        int_vec,
        schema = na_type(parsed$storage_type)
      )
      nanoarrow_array_set_schema(storage, schema)
      storage
    },
    date64 = {
      storage <- as_nanoarrow_array(
        as.numeric(x) * 86400000,
        schema = na_type(parsed$storage_type)
      )
      nanoarrow_array_set_schema(storage, schema)
      storage
    },
    NextMethod()
  )
}

#' @export
as_nanoarrow_array.POSIXlt <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  as_nanoarrow_array(new_data_frame(x, length(x)), schema = schema)
}

#' @export
as_nanoarrow_array.factor <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  if (is.null(schema$dictionary)) {
    return(as_nanoarrow_array(as.character(x), schema = schema))
  }

  storage <- schema
  storage$dictionary <- NULL

  array <- as_nanoarrow_array(unclass(x) - 1L, schema = storage)
  array$dictionary <- as_nanoarrow_array(levels(x), schema = schema$dictionary)
  array
}

#' @export
as_nanoarrow_array.vctrs_unspecified <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  } else {
    schema <- as_nanoarrow_schema(schema)
  }

  schema <- as_nanoarrow_schema(schema)
  parsed <- nanoarrow_schema_parse(schema)
  if (!is.null(parsed$extension_name)) {
    spec <- resolve_nanoarrow_extension(parsed$extension_name)
    return(as_nanoarrow_array_extension(spec, x, ..., schema = schema))
  }

  switch(
    parsed$storage_type,
    na = {
      array <- nanoarrow_array_init(schema)
      array$length <- length(x)
      array$null_count <- length(x)
      array
    },
    NextMethod()
  )
}

# Called from C to create a union array when requested.
# There are other types of objects that might make sense to
# convert to a union but we basically just need enough to
# for testing at this point.
union_array_from_data_frame <- function(x, schema) {
  if (length(x) == 0 || length(x) > 127) {
    stop(
      sprintf(
        "Can't convert data frame with %d columns to union array",
        length(x)
      )
    )
  }

  # Compute NAs
  x_is_na <- do.call("cbind", lapply(x, is.na))

  # Make sure we only have one non-NA value per row to make sure we don't drop
  # values
  stopifnot(all(rowSums(!x_is_na) <= 1))

  child_index <- rep_len(0L, nrow(x))
  seq_x <- seq_along(x)
  for (i in seq_along(child_index)) {
    for (j in seq_x) {
      if (!x_is_na[i, j]) {
        child_index[i] <- j - 1L
        break;
      }
    }
  }

  switch(
    nanoarrow_schema_parse(schema)$storage_type,
    "dense_union" = {
      is_child <- lapply(seq_x - 1L, "==", child_index)
      child_offset_each <- lapply(is_child, function(x) cumsum(x) - 1L)
      child_offset <- lapply(seq_along(child_index), function(i) {
        child_offset_each[[child_index[i] + 1]][i]
      })

      children <- Map("[", x, is_child, drop = FALSE)
      names(children) <- names(schema$children)
      array <- nanoarrow_array_init(schema)
      nanoarrow_array_modify(
        array,
        list(
          length = length(child_index),
          null_count = 0,
          buffers = list(as.raw(child_index), as.integer(child_offset)),
          children = children
        )
      )
    },
    "sparse_union" = {
      struct_schema <- na_struct(schema$children)
      array <- as_nanoarrow_array(x, array = struct_schema)
      array <- nanoarrow_array_modify(
        array,
        list(buffers = list(as.raw(child_index))),
        validate = FALSE
      )
      nanoarrow_array_set_schema(array, schema, validate = TRUE)
      array
    },
    stop("Attempt to create union from non-union array type")
  )
}

# This is defined because it's verbose to pass named arguments from C.
# When converting data frame columns, we try the internal C conversions
# first to save R evaluation overhead. When the internal conversions fail,
# we call as_nanoarrow_array() to dispatch to conversions defined via S3
# dispatch, making sure to let the default method know that we've already
# tried the internal C conversions.
as_nanoarrow_array_from_c <- function(x, schema) {
  result <- as_nanoarrow_array(x, schema = schema, .from_c = TRUE)

  # Anything we get from an S3 method we need to validate (even from the
  # arrow package, which occasionally does not honour the schema argument)
  nanoarrow_array_set_schema(result, schema, validate = TRUE)

  result
}

# Helper to allow us to use nanoarrow's string parser, which parses integers
# to set decimal storage but not the slightly more useful case of parsing
# things with decimal points yet.
storage_integer_for_decimal <- function(numbers, scale) {
  rounded_formatted <- storage_decimal_for_decimal(numbers, scale)
  gsub(".", "", rounded_formatted, fixed = TRUE)
}

storage_decimal_for_decimal <- function(numbers, scale) {
  if (scale > 0) {
    rounded_formatted <- sprintf("%0.*f", scale, numbers)
    rounded_formatted[is.na(numbers)] <- NA_character_
    rounded_formatted
  } else {
    rounded_formatted <- as.character(round(numbers, scale))
    gsub(paste0("0{", -scale, "}$"), "", rounded_formatted)
  }
}
