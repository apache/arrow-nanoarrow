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
as_nanoarrow_array.POSIXct <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  parsed <- nanoarrow_schema_parse(schema)
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

  parsed <- nanoarrow_schema_parse(schema)
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
as_nanoarrow_array.Date <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
  }

  parsed <- nanoarrow_schema_parse(schema)
  switch(
    parsed$type,
    date32 = {
      storage <- as_nanoarrow_array(
        as.integer(x),
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

  as_nanoarrow_array(new_data_frame(x, length(x)), schema = schema)
}

#' @export
as_nanoarrow_array.factor <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(x)
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

  switch(
    nanoarrow_schema_parse(schema)$storage_type,
    na = {
      array <- nanoarrow_array_init(schema)
      array$length <- length(x)
      array$null_count <- length(x)
      array
    },
    NextMethod()
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
