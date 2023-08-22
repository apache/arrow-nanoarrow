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
#' - [Date][as.Date]: Only the date32 type can be converted to an R Date vector.
#' - [hms::hms()]: Time32 and time64 types can be converted to [hms::hms()].
#' - [difftime()]: Time32, time64, and duration types can be converted to
#'   R [difftime()] vectors. The value is converted to match the [units()]
#'   attribute of `to`.
#' - [blob::blob()]: String, large string, binary, and large binary types can
#'   be converted to [blob::blob()].
#' - [vctrs::list_of()]: List, large list, and fixed-size list types can be
#'   converted to [vctrs::list_of()].
#' - [data.frame()]: Struct types can be converted to [data.frame()].
#' - [vctrs::unspecified()]: Any type can be converted to [vctrs::unspecified()];
#'   however, a warning will be raised if any non-null values are encountered.
#'
#' In addition to the above conversions, a null array may be converted to any
#' target prototype except [data.frame()]. Extension arrays are currently
#' converted as their storage type; dictionary-encoded arrays are not
#' currently supported.
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
convert_array_from_c <- function(array, to) {
  convert_array(array, to, .from_c = TRUE)
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
  levels_final <- levels(to)
  levels <- convert_array(array$dictionary, character())
  array$dictionary <- NULL
  indices <- convert_array(array, integer()) + 1L

  if (identical(levels, levels_final)) {
    structure(indices, levels = levels_final, class = "factor")
  } else if (all(levels %in% levels_final)) {
    level_map <- match(levels, levels_final)
    structure(level_map[indices], levels = levels_final, class = "factor")
  } else {
    stop("Error converting to factor: some levels in data do not exist in levels")
  }
}

#' @export
convert_array.vctrs_partial_factor <- function(array, to, ...) {
  if (!identical(to$partial, factor())) {
    stop_cant_convert_array(array, to)
  }

  levels <- convert_array(array$dictionary, character())
  array$dictionary <- NULL
  indices <- convert_array(array, integer()) + 1L
  structure(indices, levels = levels, class = "factor")
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

# These are conversions that are called from low-level materializer
# C functions. They have to be called in this way because the destination
# (i.e. `to`) is handled mostly in C (i.e., we can't use an S3 generic
# since that would apply to *all* conversions of that type and we
# definitely don't want S3 dispatch overhead for things like double() and
# character())

# Used for decimal128 -> double conversion
convert_fallback_arrow <- function(array, schema, offset, length, args) {
  assert_arrow_installed(
    sprintf(
      "convert %s array to object of type double",
      nanoarrow_schema_formatted(schema)
    )
  )

  # Because we are passing to arrow and arrow will release the C structure,
  # we need to export it. Doing this by hand to minimize overhead.
  array2 <- nanoarrow_allocate_array()
  schema2 <- nanoarrow_allocate_schema()
  nanoarrow_pointer_export(array, array2)
  nanoarrow_pointer_export(schema, schema2)
  arrow_array <- arrow::Array$import_from_c(array2, schema2)
  arrow_array$Slice(offset, length)$as_vector()
}

# Used for dictionary<string> -> character()
convert_fallback_dictionary_chr <- function(array, schema, offset, length, args) {
  values <- .Call(nanoarrow_c_convert_array, array$dictionary, character())
  array$dictionary <- NULL
  indices <- .Call(nanoarrow_c_convert_array, array, integer())
  values[indices + 1L]
}

# Called from C for conversions that are not handled there (e.g.,
# decimal, dictionary, extension)
convert_fallback_other <- function(array, schema, offset, length, args) {
  to <- args[[1]]

  # Ensures we have a modifiable shallow copy on hand with the correct
  # offset/length.
  array <- nanoarrow_array_modify(
    array,
    list(offset = offset, length = length),
    validate = FALSE
  )

  # Call convert_array() on a single chunk. Use .from_c = TRUE to ensure that
  # methods do not attempt to pass the same array back to the C conversions.
  # When the result is passed back to C it is checked enough to avoid segfault
  # but not necessarily for correctness (e.g., factors with levels that don't
  # correspond to 'to').
  convert_array(array, to, .from_c = TRUE)
}
