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

#' @importFrom utils str
#' @export
str.nanoarrow_buffer <- function(object, ...) {
  cat(sprintf("%s\n", format(object)))
  invisible(object)
}

#' @export
print.nanoarrow_buffer <- function(x, ...) {
  str(x, ...)
  invisible(x)
}

#' @export
format.nanoarrow_buffer <- function(x, ...) {
  info <- .Call(nanoarrow_c_buffer_info, x)
  size_bytes <- info$size_bytes %||% NA_integer_
  sprintf(
    "<%s[%s b] at %s>",
    class(x)[1],
    size_bytes,
    nanoarrow_pointer_addr_pretty(x)
  )
}

#' @export
as.raw.nanoarrow_buffer <- function(x, ...) {
  .Call(nanoarrow_c_buffer_as_raw, x)
}
