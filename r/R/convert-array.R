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
#' @param array A [nanoarrow_array][as_nanoarrow_array].
#' @param to A target prototype object describing the type to which `array`
#'   should be converted, or `NULL` to use the default conversion.
#' @param ... Passed to S3 methods
#'
#' @return An R vector of type `to`.
#' @export
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
