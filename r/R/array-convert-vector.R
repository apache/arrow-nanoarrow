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
#' @param array A [nanoarrow_array][as_nanoarrow_array].
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
  .Call(nanoarrow_c_from_array, array, to)
}

#' @export
from_nanoarrow_array.vctrs_partial_frame <- function(array, to, ...) {
  ptype <- infer_nanoarrow_ptype(array)
  ptype <- vctrs::vec_cast(ptype, to)
  .Call(nanoarrow_c_from_array, array, ptype)
}

#' @rdname from_nanoarrow_array
#' @export
infer_nanoarrow_ptype <- function(array, ...) {
  .Call(nanoarrow_c_infer_ptype, array)
}
