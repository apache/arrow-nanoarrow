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

#' Convert an object to a nanoarrow_buffer
#'
#' @param x An object to convert to a buffer
#' @param ... Passed to S3 methods
#'
#' @return An object of class 'nanoarrow_buffer'
#' @export
#'
#' @examples
#' array <- as_nanoarrow_array(1:4)
#' array$buffers[[2]]
#' as.raw(array$buffers[[2]])
#'
as_nanoarrow_buffer <- function(x, ...) {
  UseMethod("as_nanoarrow_buffer")
}

#' @export
as_nanoarrow_buffer.nanoarrow_buffer <- function(x, ...) {
  x
}

#' @export
as_nanoarrow_buffer.default <- function(x, ...) {
  result <- tryCatch(
    .Call(nanoarrow_c_as_buffer_default, x),
    error = function(...) NULL
  )

  if (is.null(result)) {
    cls <- paste(class(x), collapse = "/")
    stop(sprintf("Can't convert object of type %s to nanoarrow_buffer", cls))
  }

  result
}

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
  info <- nanoarrow_buffer_info(x)
  size_bytes <- info$size_bytes %||% NA_integer_
  sprintf(
    "<%s[%s b] at %s>",
    class(x)[1],
    size_bytes,
    nanoarrow_pointer_addr_pretty(info$data)
  )
}

#' @export
as.raw.nanoarrow_buffer <- function(x, ...) {
  .Call(nanoarrow_c_buffer_as_raw, x)
}

nanoarrow_buffer_info <- function(x) {
  .Call(nanoarrow_c_buffer_info, x)
}
