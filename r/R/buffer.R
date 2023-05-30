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

#' Convert an object to a nanoarrow buffer
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
#' as_nanoarrow_buffer(1:5)
#'
#' buffer <- as_nanoarrow_buffer(NULL)
#'
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
  if (info$data_type == "unknown") {
    len <- ""
  } else if (info$element_size_bits == 0 || info$data_type %in% c("binary", "string")) {
    len <- sprintf("[%s b]", info$size_bytes)
  } else {
    logical_length <- (info$size_bytes * 8) %/% info$element_size_bits
    len <- sprintf("[%s][%s b]", logical_length, info$size_bytes)
  }


  sprintf(
    "<%s %s<%s>%s at %s>",
    class(x)[1],
    info$type,
    info$data_type,
    len,
    nanoarrow_pointer_addr_pretty(info$data)
  )
}

#' Create and modify nanoarrow buffers
#'
#' @param buffer,new_buffer [nanoarrow_buffer][as_nanoarrow_buffer]s.
#' @inheritParams convert_array
#'
#' @return
#'   - `nanoarrow_buffer_init()`: An object of class 'nanoarrow_buffer'
#'   - `nanoarrow_buffer_append()`: Returns `buffer`, invisibly. Note that
#'     `buffer` is modified in place by reference.
#' @export
#'
#' @examples
#' buffer <- nanoarrow_buffer_init()
#' nanoarrow_buffer_append(buffer, 1:5)
#'
#' array <- nanoarrow_array_modify(
#'   nanoarrow_array_init(na_int32()),
#'   list(length = 5, buffers = list(NULL, buffer))
#' )
#' as.vector(array)
#'
nanoarrow_buffer_init <- function() {
  as_nanoarrow_buffer(NULL)
}

#' @rdname nanoarrow_buffer_init
#' @export
nanoarrow_buffer_append <- function(buffer, new_buffer) {
  buffer <- as_nanoarrow_buffer(buffer)
  new_buffer <- as_nanoarrow_buffer(new_buffer)

  .Call(nanoarrow_c_buffer_append, buffer, new_buffer)

  invisible(buffer)
}

#' @rdname nanoarrow_buffer_init
#' @export
convert_buffer <- function(buffer, to = NULL) {
  convert_array(as_nanoarrow_array.nanoarrow_buffer(buffer), to = to)
}

#' @export
as_nanoarrow_array.nanoarrow_buffer <- function(x, ..., schema = NULL) {
  if (!is.null(schema)) {
    stop("as_nanoarrow_array(<nanoarow_buffer>) with non-NULL schema is not supported")
  }

  info <- nanoarrow_buffer_info(x)
  if (info$data_type == "unknown" || info$element_size_bits == 0) {
    stop("Can't convert buffer with unknown type or unknown element size")
  }

  data_type <- info$data_type
  logical_length <- (info$size_bytes * 8) %/% info$element_size_bits

  if (data_type %in% c("string", "binary") && logical_length <= .Machine$integer.max) {
    array <- nanoarrow_array_init(na_type(data_type))
    offsets <- as.integer(c(0, logical_length))
    nanoarrow_array_modify(
      array,
      list(
        length = 1,
        null_count = 0,
        buffers = list(NULL, offsets, x)
      )
    )
  } else if(data_type %in% c("string", "binary")) {
    array <- nanoarrow_array_init(na_type(paste0("large_", data_type)))
    offsets <- as_nanoarrow_array(c(0, logical_length), schema = na_int64())$buffers[[2]]
    nanoarrow_array_modify(
      array,
      list(
        length = 1,
        null_count = 0,
        buffers = list(NULL, offsets, x)
      )
    )
  } else {
    array <- nanoarrow_array_init(na_type(data_type))
    nanoarrow_array_modify(
      array,
      list(
        length = logical_length,
        null_count = 0,
        buffers = list(NULL, x)
      )
    )
  }
}

#' @export
as.raw.nanoarrow_buffer <- function(x, ...) {
  .Call(nanoarrow_c_buffer_as_raw, x)
}

nanoarrow_buffer_info <- function(x) {
  .Call(nanoarrow_c_buffer_info, x)
}


# This is the list()-like interface to nanoarrow_buffer that allows $ and [[
# to make nice auto-complete when interacting in an IDE

#' @export
length.nanoarrow_buffer <- function(x, ...) {
  5L
}

#' @export
names.nanoarrow_buffer <- function(x, ...) {
  c("data", "size_bytes", "capacity_bytes", "type", "data_type", "element_size_bits")
}

#' @export
`[[.nanoarrow_buffer` <- function(x, i, ...) {
  nanoarrow_buffer_info(x)[[i]]
}

#' @export
`$.nanoarrow_buffer` <- function(x, i, ...) {
  x[[i]]
}
