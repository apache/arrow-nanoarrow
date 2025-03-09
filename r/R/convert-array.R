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


#' Convert an Array into an R vector
#'
#' Converts `array` to the type specified by `to`. This is a low-level interface;
#' most users should use `as.data.frame()` or `as.vector()` unless finer-grained
#' control is needed over the conversion. This function is an S3 generic
#' dispatching on `to`: developers may implement their own S3 methods for
#' custom vector types.
#'
#' Note that unregistered extension types will by default issue a warning.
#' Use `options(nanoarrow.warn_unregistered_extension = FALSE)` to disable
#' this behaviour.
#'
#' @param array A [nanoarrow_array][as_nanoarrow_array].
#' @param to A target prototype object describing the type to which `array`
#'   should be converted, or `NULL` to use the default conversion as
#'   returned by [infer_nanoarrow_ptype()]. Alternatively, a function can be
#'   passed to perform an alternative calculation of the default ptype as
#'   a function of `array` and the default inference of the prototype.
#' @param ... Passed to S3 methods
#'
#' @return An R vector of type `to`.
#' @export
#'
#' @details
#' Conversions are implemented for the following R vector types:
#'
#' - [logical()]: Any numeric type can be converted to [logical()] in addition
#'   to the bool type. For numeric types, any non-zero value is considered `TRUE`.
#' - [integer()]: Any numeric type can be converted to [integer()]; however,
#'   a warning will be signaled if the any value is outside the range of the
#'   32-bit integer.
#' - [double()]: Any numeric type can be converted to [double()]. This
#'   conversion currently does not warn for values that may not roundtrip
#'   through a floating-point double (e.g., very large uint64 and int64 values).
#' - [character()]: String and large string types can be converted to
#'   [character()]. The conversion does not check for valid UTF-8: if you need
#'   finer-grained control over encodings, use `to = blob::blob()`.
#' - [factor()]: Dictionary-encoded arrays of strings can be converted to
#'   `factor()`; however, this must be specified explicitly (i.e.,
#'   `convert_array(array, factor())`) because arrays arriving
#'   in chunks can have dictionaries that contain different levels. Use
#'   `convert_array(array, factor(levels = c(...)))` to materialize an array
#'   into a vector with known levels.
#' - [Date][as.Date()]: Only the date32 type can be converted to an R Date vector.
#' - [hms::hms()]: Time32 and time64 types can be converted to [hms::hms()].
#' - [difftime()]: Time32, time64, and duration types can be converted to
#'   R [difftime()] vectors. The value is converted to match the [units()]
#'   attribute of `to`.
#' - [blob::blob()]: String, large string, binary, and large binary types can
#'   be converted to [blob::blob()].
#' - [vctrs::list_of()]: List, large list, and fixed-size list types can be
#'   converted to [vctrs::list_of()].
#' - [matrix()]: Fixed-size list types can be converted to
#'   `matrix(ptype, ncol = fixed_size)`.
#' - [data.frame()]: Struct types can be converted to [data.frame()].
#' - [vctrs::unspecified()]: Any type can be converted to [vctrs::unspecified()];
#'   however, a warning will be raised if any non-null values are encountered.
#'
#' In addition to the above conversions, a null array may be converted to any
#' target prototype except [data.frame()]. Extension arrays are currently
#' converted as their storage type.
#'
#' @examples
#' array <- as_nanoarrow_array(data.frame(x = 1:5))
#' str(convert_array(array))
#' str(convert_array(array, to = data.frame(x = double())))
#'
convert_array <- function(array, to = NULL, ...) {
  stopifnot(inherits(array, "nanoarrow_array"))
  UseMethod("convert_array", to)
}

#' @export
convert_array.default <- function(array, to = NULL, ..., .from_c = FALSE) {
  if (.from_c) {
    # Handle extension conversion
    # We don't need the user-friendly versions and this is performance-sensitive
    schema <- .Call(nanoarrow_c_infer_schema_array, array)
    parsed <- .Call(nanoarrow_c_schema_parse, schema)
    if (!is.null(parsed$extension_name)) {
      spec <- resolve_nanoarrow_extension(parsed$extension_name)
      return(convert_array_extension(spec, array, to, ...))
    }

    # Handle default dictionary conversion since it's the same for all types
    dictionary <- array$dictionary

    if (!is.null(dictionary)) {
      values <- .Call(nanoarrow_c_convert_array, dictionary, to)
      array$dictionary <- NULL
      indices <- .Call(nanoarrow_c_convert_array, array, integer())
      return(vec_slice2(values, indices + 1L))
    }

    stop_cant_convert_array(array, to)
  }

  if (is.function(to)) {
    to <- to(array, infer_nanoarrow_ptype(array))
  }

  .Call(nanoarrow_c_convert_array, array, to)
}

# This is defined because it's verbose to pass named arguments from C.
# When converting data frame columns, we try the internal C conversions
# first to save R evaluation overhead. When the internal conversions fail,
# we call convert_array() to dispatch to conversions defined via S3
# dispatch, making sure to let the default method know that we've already
# tried the internal C conversions.
convert_fallback_other <- function(array, offset, length, to) {
  # If we need to modify offset/length, do it using a shallow copy.
  if (!is.null(offset)) {
    array <- nanoarrow_array_modify(
      array,
      list(offset = offset, length = length),
      validate = FALSE
    )
  }

  # Call convert_array() on a single chunk. Use .from_c = TRUE to ensure that
  # methods do not attempt to pass the same array back to the C conversions.
  # When the result is passed back to C it is checked enough to avoid segfault
  # but not necessarily for correctness (e.g., factors with levels that don't
  # correspond to 'to'). This result may be used as-is or may be copied into
  # a slice of another vector.
  convert_array(array, to, .from_c = TRUE)
}

#' @export
convert_array.nanoarrow_vctr <- function(array, to, ...) {
  schema <- attr(to, "schema", exact = TRUE)
  if (is.null(schema)) {
    schema <- infer_nanoarrow_schema(array)
  }

  new_nanoarrow_vctr(list(array), schema, class(to))
}

#' @export
convert_array.double <- function(array, to, ...) {
  # Handle conversion from decimal128 via arrow
  schema <- infer_nanoarrow_schema(array)
  parsed <- nanoarrow_schema_parse(schema)
  if (parsed$type == "decimal128") {
    # assert_arrow_installed(
    #   sprintf(
    #     "convert %s array to object of type double",
    #     nanoarrow_schema_formatted(schema)
    #   )
    # )
    #
    # arrow_array <- as_arrow_array.nanoarrow_array(array)
    # arrow_array$as_vector()
    NextMethod()
  } else {
    NextMethod()
  }
}

#' @export
convert_array.vctrs_partial_frame <- function(array, to, ...) {
  ptype <- infer_nanoarrow_ptype(array)
  if (!is.data.frame(ptype)) {
    stop_cant_convert_array(array, to)
  }

  ptype <- vctrs::vec_ptype_common(ptype, to)
  .Call(nanoarrow_c_convert_array, array, ptype)
}

#' @export
convert_array.factor <- function(array, to, ...) {
  if (!is.null(array$dictionary)) {
    levels_final <- levels(to)
    levels <- convert_array(array$dictionary, character())
    array$dictionary <- NULL
    indices <- convert_array(array, integer()) + 1L

    # Handle empty factor() as the sentinel for "auto levels"
    if (identical(levels(to), character())) {
      levels(to) <- levels
    }

    if (identical(levels, levels(to))) {
      fct_data <- indices
    } else if (all(levels %in% levels(to))) {
      level_map <- match(levels, levels(to))
      fct_data <- level_map[indices]
    } else {
      stop("Error converting to factor: some levels in data do not exist in levels")
    }
  } else {
    strings <- convert_array(array, character())

    # Handle empty factor() as the sentinel for "auto levels"
    if (identical(levels(to), character())) {
      fct_data <- factor(strings, levels)
      levels(to) <- levels(fct_data)
    } else {
      fct_data <- factor(strings, levels = levels(to))
    }
  }

  # Restore other attributes (e.g., ordered, labels)
  attributes(fct_data) <- attributes(to)
  fct_data
}

stop_cant_convert_array <- function(array, to, n = 0) {
  stop_cant_convert_schema(infer_nanoarrow_schema(array), to, n - 1)
}

stop_cant_convert_schema <- function(schema, to, n = 0) {
  schema_label <- nanoarrow_schema_formatted(schema)

  if (is.null(schema$name) || identical(schema$name, "")) {
    cnd <- simpleError(
      sprintf(
        "Can't convert array <%s> to R vector of type %s",
        schema_label,
        class(to)[1]
      ),
      call = sys.call(n - 1)
    )
  } else {
    cnd <- simpleError(
      sprintf(
        "Can't convert `%s` <%s> to R vector of type %s",
        schema$name,
        schema_label,
        class(to)[1]
      ),
      call = sys.call(n - 1)
    )
  }

  stop(cnd)
}
