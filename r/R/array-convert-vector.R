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
