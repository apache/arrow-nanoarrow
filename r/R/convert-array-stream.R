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
    # The underlying nanoarrow_c_convert_array_stream() currently requires that
    # the total length of all batches is known in advance. If the caller
    # provided this we can save a bit of work.
    return(.Call(
      nanoarrow_c_convert_array_stream,
      array_stream,
      to,
      as.double(size)[1],
      n
    ))
  }

  # Otherwise, we need to collect all batches.
  batches <- collect_array_stream(
    array_stream,
    n,
    schema = schema,
    validate = FALSE
  )

  # If 0 batches, return zero-length result consistent with 'to'
  if (length(batches) == 0L) {
    basic_stream <- .Call(nanoarrow_c_basic_array_stream, list(), schema, FALSE)
    return(.Call(nanoarrow_c_convert_array_stream, basic_stream, to, 0, Inf))
  }

  # If there is exactly one batch, use convert_array(). Converting a single
  # array currently takes a more efficient code path for types that can be
  # converted as ALTREP (e.g., strings).
  if (length(batches) == 1L) {
    return(.Call(nanoarrow_c_convert_array, batches[[1]], to))
  }

  # For `to` with known bind function, convert from already collected array and avoid recreating
  # another array stream which has very bad exponential time behavior.
  if (is.data.frame(to) || is.matrix(to)) {
    converted_batches <- lapply(batches, function(b) .Call(nanoarrow_c_convert_array, b, to))
    if (is.data.frame(to)) {
      # Prefer vctrs::vec_rbind over base::rbind because it's faster.
      converted <- do.call(vctrs::vec_rbind, converted_batches)
    } else if (is.matrix(to)) {
      converted <- do.call(rbind, converted_batches)
    } else {
      stop("unhandled case")
    }
    return(converted)
  } else {
    # When there's no way to bind converted batches together, have to create a single stream and convert that.
    # This method has horrible exponential time when ALTREP is involved, so avoid whenever possible.
    size <- .Call(nanoarrow_c_array_list_total_length, batches)
    basic_stream <- .Call(nanoarrow_c_basic_array_stream, batches, schema, FALSE)

    return(.Call(
      nanoarrow_c_convert_array_stream,
      basic_stream,
      to,
      as.double(size),
      Inf
    ))
  }
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
