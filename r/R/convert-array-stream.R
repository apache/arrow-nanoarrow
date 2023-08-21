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
#' @inheritParams basic_array_stream
#'
#' @return
#'   - `convert_array_stream()`: An R vector of type `to`.
#'   - `collect_array_stream()`: A `list()` of [nanoarrow_array][as_nanoarrow_array]
#' @export
#'
#' @examples
#' stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
#' str(convert_array_stream(stream))
#' str(convert_array_stream(stream, to = data.frame(x = double())))
#'
#' stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
#' collect_array_stream(stream)
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

  # Collect all batches into a list()
  batches <- collect_array_stream(
    array_stream,
    n,
    schema = schema,
    validate = FALSE
  )

  # If there is exactly one batch, use convert_array()
  if (length(batches) == 1L) {
    return(.Call(nanoarrow_c_convert_array, batches[[1]], to))
  }

  # Otherwise, compute the final size, create another array stream,
  # and call convert_array_stream() with a known size.
  lengths <- vapply(
    batches,
    # Use custom accessor because array$length in a loop is slow
    function(array) {
      .Call(nanoarrow_c_array_proxy, array, NULL, FALSE)$length
    },
    double(1)
  )
  basic_stream <- .Call(nanoarrow_c_basic_array_stream, batches, schema, FALSE)
  convert_array_stream(basic_stream, to = to, size = sum(lengths))
}

#' @rdname convert_array_stream
#' @export
collect_array_stream <- function(array_stream, n = Inf, schema = NULL,
                                 validate = TRUE) {
  stopifnot(
    inherits(array_stream, "nanoarrow_array_stream")
  )

  if (is.null(schema)) {
    schema <- .Call(nanoarrow_c_array_stream_get_schema, array_stream)
  }

  batches <- vector("list", 1024L)
  n_batches <- 0
  get_next <- array_stream$get_next
  while (n_batches < n) {
    array <- get_next(schema, validate = validate)
    if (is.null(array)) {
      break
    }

    n_batches <- n_batches + 1

    # This assignment has reasonable (but not great) performance when
    # n_batches > 1024 in recent versions of R because R overallocates vectors
    # slightly to support this pattern. It may be worth moving this
    # implementation to C or C++ in the future if the collect step becomes a
    # bottleneck.
    batches[[n_batches]] <- array
  }

  batches[seq_len(n_batches)]
}
