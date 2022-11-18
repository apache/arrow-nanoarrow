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

#' Convert an Array Stream into an R vector
#'
#' Converts `array_stream` to the type specified by `to`. This is a low-level
#' interface; most users should use `as.data.frame()` or `as.vector()` unless
#' finer-grained control is needed over the conversion. See [convert_array()]
#' for details of the conversion process; see [infer_nanoarrow_ptype()] for
#' default inferences of `to`.
#'
#' @param array_stream A [nanoarrow_array_stream][as_nanoarrow_array_stream].
#' @param size The exact size of the output, if known. If specified,
#'   slightly more efficient implementation may be used to collect the output.
#' @param n The maximum number of batches to pull from the array stream.
#' @inheritParams convert_array
#'
#' @return An R vector of type `to`.
#' @export
#'
#' @examples
#' stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
#' str(convert_array_stream(stream))
#' str(convert_array_stream(stream, to = data.frame(x = double())))
#'
convert_array_stream <- function(array_stream, to = NULL, size = NULL, n = Inf) {
  stopifnot(
    inherits(array_stream, "nanoarrow_array_stream")
  )

  schema <- .Call(nanoarrow_c_array_stream_get_schema, array_stream)
  if (is.null(to)) {
    to <- infer_nanoarrow_ptype(schema)
  } else if (is.function(to)) {
    to <- to(schema, infer_nanoarrow_ptype(schema))
  }

  n <- as.double(n)[1]
  if (!is.null(size)) {
    return(
      .Call(
        nanoarrow_c_convert_array_stream,
        array_stream,
        to,
        as.double(size)[1],
        n
      )
    )
  }

  batches <- vector("list", 1024L)
  n_batches <- 0L
  get_next <- array_stream$get_next
  while (!is.null(array <- get_next(schema, validate = FALSE)) && (n_batches < n)) {
    n_batches <- n_batches + 1L
    batches[[n_batches]] <- .Call(nanoarrow_c_convert_array, array, to)
  }

  if (n_batches == 0L && is.data.frame(to)) {
    to[integer(0), , drop = FALSE]
  } else if (n_batches == 0L && is.data.frame(to)) {
    to[integer(0)]
  } else if (n_batches == 1L) {
    batches[[1]]
  } else if (inherits(to, "data.frame")) {
    do.call(rbind, batches[seq_len(n_batches)])
  } else {
    do.call(c, batches[seq_len(n_batches)])
  }
}
