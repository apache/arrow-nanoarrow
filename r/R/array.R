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
#' @param x An object to convert to a array
#' @param schema An optional schema used to enforce conversion to a particular
#'   type. Defaults to [infer_nanoarrow_schema()].
#' @param to A target prototype object describing the type to which `array`
#'   should be converted.
#' @param array An object of class 'nanoarrow_array'
#' @param ... Passed to S3 methods
#'
#' @return An object of class 'nanoarrow_array'
#' @export
as_nanoarrow_array <- function(x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array")
}

#' @rdname as_nanoarrow_array
#' @export
from_nanoarrow_array <- function(array, to = NULL, ...) {
  stopifnot(inherits(array, "nanoarrow_array"))
  UseMethod("from_nanoarrow_array", to)
}

#' @export
as.vector.nanoarrow_array <- function(x, mode = "any") {
  stopifnot(identical(mode, "any"))
  from_nanoarrow_array(x)
}

#' @export
as.data.frame.nanoarrow_array <- function(x, ...) {
  from_nanoarrow_array(x, to = vctrs::partial_frame())
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
from_nanoarrow_array.default <- function(array, to = NULL, ...) {
  # For now, use arrow's conversion for everything
  result <- as.vector(arrow::as_arrow_array(array))

  # arrow's conversion doesn't support `to`, so for now use an R cast
  # workaround for a bug in vctrs: https://github.com/r-lib/vctrs/issues/1642
  if (inherits(result, "tbl_df")) {
    result <- new_data_frame(result, nrow(result))
  }

  vctrs::vec_cast(result, to)
}

#' @export
infer_nanoarrow_schema.nanoarrow_array <- function(x, ...) {
  .Call(nanoarrow_c_infer_schema_array, x) %||%
    stop("nanoarrow_array() has no associated schema")
}

nanoarrow_array_set_schema <- function(array, schema) {
  .Call(nanoarrow_c_array_set_schema, array, schema)
  invisible(array)
}
