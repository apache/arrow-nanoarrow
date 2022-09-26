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


#' Convert an Array to an R vector
#'
#' @param array An object of class 'nanoarrow_array'
#' @param schema A [nanoarrow schema][as_nanoarrow_schema] to use as a target
#' @param to A target prototype object describing the type to which `array`
#'   should be converted, or `NULL` to use the default conversion.
#' @param ... Passed to S3 methods
#'
#' @return An R vector of type `to`.
#' @export
#'
from_nanoarrow_array <- function(array, to = NULL, ...) {
  stopifnot(inherits(array, "nanoarrow_array"))
  UseMethod("from_nanoarrow_array", to)
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

#' @rdname from_nanoarrow_array
#' @export
infer_nanoarrow_ptype <- function(schema, ...) {
  # For now, just convert a zero-size arrow array to a vector
  # and see what we get
  as.vector(arrow::concat_arrays(type = arrow::as_data_type(schema)))
}

#' @export
from_nanoarrow_array.vctrs_partial_frame <- function(array, to, ...) {
  nrows <- array$length
  children <- lapply(array$children, as.vector)
  new_data_frame(children, nrows)
}

#' @export
from_nanoarrow_array.data.frame <- function(array, to, ...) {
  nrows <- array$length
  children <- Map(from_nanoarrow_array, array$children, to)
  names(children) <- names(to)
  result <- new_data_frame(children, nrows)
  class(result) <- class(to)
  result
}

#' @export
from_nanoarrow_array.character <- function(array, to, ...) {
  nanoarrow_altrep(array, to) %||%
    stop("Can't convert array to character()")
}
