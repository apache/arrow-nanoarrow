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

#' Experimental Arrow encoded arrays as R vectors
#'
#' @param x An object that works with [as_nanoarrow_array_stream()]. Most
#'   spatial objects in R already work with this method.
#' @param ... Passed to [as_nanoarrow_array_stream()]
#' @param schema An optional `schema`
#'
#' @return A vctr of class 'nanoarrow_vctr'
#' @export
#'
#' @examples
#' array <- as_nanoarrow_array(1:5)
#' as_nanoarrow_vctr(array)
#'
as_nanoarrow_vctr <- function(x, ..., schema = NULL) {
  if (inherits(x, "nanoarrow_vctr") && is.null(schema)) {
    return(x)
  }

  stream <- as_nanoarrow_array_stream(x, ..., schema = schema)
  chunks <- collect_array_stream(stream, validate = FALSE)
  new_nanoarrow_vctr(chunks, stream$get_schema())
}

#' @rdname as_nanoarrow_vctr
#' @export
nanoarrow_vctr <- function(schema = NULL) {
  if (is.null(schema)) {
    new_nanoarrow_vctr(list(), NULL)
  } else {
    new_nanoarrow_vctr(list(), as_nanoarrow_schema(schema))
  }
}

new_nanoarrow_vctr <- function(chunks, schema, indices = NULL) {
  offsets <- .Call(nanoarrow_c_vctr_chunk_offsets, chunks)
  if (is.null(indices)) {
    indices <- seq_len(offsets[length(offsets)])
  }

  structure(
    indices,
    schema = schema,
    chunks = chunks,
    offsets = offsets,
    class = "nanoarrow_vctr"
  )
}

#' @export
`[.nanoarrow_vctr` <- function(x, i) {
  attrs <- attributes(x)
  x <- NextMethod()

  if (is.null(vctr_as_slice(x))) {
    stop(
      "Can't subset nanoarrow_vctr with non-slice (e.g., only i:j indexing is supported)"
    )
  }

  attributes(x) <- attrs
  x
}

#' @export
`[<-.nanoarrow_vctr` <- function(x, i, value) {
  stop("subset assignment for nanoarrow_vctr is not supported")
}

#' @export
`[[<-.nanoarrow_vctr` <- function(x, i, value) {
  stop("subset assignment for nanoarrow_vctr is not supported")
}

#' @export
format.nanoarrow_vctr <- function(x, ...) {
  # Technically we can do better here
  stream <- as_nanoarrow_array_stream(x)
  format(convert_array_stream(stream), ...)
}

# Because RStudio's viewer uses this, we want to use the potentially abbreviated
# format string.
#' @export
as.character.nanoarrow_vctr <- function(x, ...) {
  format(x, ...)
}

#' @export
infer_nanoarrow_schema.nanoarrow_vctr <- function(x, ...) {
  attr(x, "schema", exact = TRUE)
}

# Because zero-length vctrs are R's way of communicating "type", implement
# as_nanoarrow_schema() here so that it works in places that expect a type
#' @export
as_nanoarrow_schema.nanoarrow_vctr <- function(x, ...) {
  attr(x, "schema", exact = TRUE)
}

#' @export
as_nanoarrow_array_stream.nanoarrow_vctr <- function(x, ..., schema = NULL) {
  as_nanoarrow_array_stream.nanoarrow_vctr(x, ..., schema = schema)
}

#' @export
as_nanoarrow_array_stream.nanoarrow_vctr <- function(x, ..., schema = NULL) {
  if (!is.null(schema)) {
    stream <- as_nanoarrow_array_stream(x, schema = NULL)
    return(as_nanoarrow_array_stream(stream, schema = schema))
  }

  slice <- vctr_as_slice(x)
  if (is.null(slice)) {
    stop("Can't resolve non-slice nanoarrow_vctr to nanoarrow_array_stream")
  }

  x_schema <- attr(x, "schema", exact = TRUE)

  # Zero-size slice can be an array stream with zero batches
  if (slice[2] == 0) {
    return(basic_array_stream(list(), schema = x_schema))
  }

  # Full slice doesn't need slicing logic
  offsets <- attr(x, "offsets", exact = TRUE)
  batches <- attr(x, "chunks", exact = TRUE)
  if (slice[1] == 1 && slice[2] == max(offsets)) {
    return(
      basic_array_stream(
        batches,
        schema = x_schema,
        validate = FALSE
      )
    )
  }

  # Calculate first and last slice information
  first_index <- slice[1] - 1L
  end_index <- first_index + slice[2]
  last_index <- end_index - 1L
  first_chunk_index <- vctr_resolve_chunk(first_index, offsets)
  last_chunk_index <- vctr_resolve_chunk(last_index, offsets)

  first_chunk_offset <- first_index - offsets[first_chunk_index + 1L]
  first_chunk_length <- offsets[first_chunk_index + 2L] - first_index
  last_chunk_offset <- 0L
  last_chunk_length <- end_index - offsets[last_chunk_index + 1L]

  # Calculate first and last slices
  if (first_chunk_index == last_chunk_index) {
    batch <- vctr_array_slice(
      batches[[first_chunk_index + 1L]],
      first_chunk_offset,
      last_chunk_length - first_chunk_offset
    )

    return(
      basic_array_stream(
        list(batch),
        schema = x_schema,
        validate = FALSE
      )
    )
  }

  batch1 <- vctr_array_slice(
    batches[[first_chunk_index + 1L]],
    first_chunk_offset,
    first_chunk_length
  )

  batchn <- vctr_array_slice(
    batches[[last_chunk_index + 1L]],
    last_chunk_offset,
    last_chunk_length
  )

  seq_mid <- seq_len(last_chunk_index - first_chunk_index - 1)
  batch_mid <- batches[first_chunk_index + seq_mid]

  basic_array_stream(
    c(
      list(batch1),
      batch_mid,
      list(batchn)
    ),
    schema = x_schema,
    validate = FALSE
  )
}

#' @export
c.nanoarrow_vctr <- function(...) {
  stop("c() not implemented for nanoarrow_vctr()")
}

# Ensures that nanoarrow_vctr can fit in a data.frame
#' @export
as.data.frame.nanoarrow_vctr <- function(x, ..., optional = FALSE) {
  if (!optional) {
    stop(sprintf("cannot coerce object of tyoe '%s' to data.frame", class(x)[1]))
  } else {
    new_data_frame(list(x))
  }
}

#' @export
print.nanoarrow_vctr <- function(x, ...) {
  schema <- attr(x, "schema", exact = TRUE)
  if (is.null(schema)) {
    cat("<nanoarrow_vctr sentinel>\n")
    return(invisible(x))
  }

  formatted <- nanoarrow_schema_formatted(schema, recursive = FALSE)
  cat(sprintf("<nanoarrow_vctr %s[%d]>\n", formatted, length(x)))

  n_values <- min(length(x), 20)
  more_values <- length(x) - n_values
  stream <- as_nanoarrow_array_stream(utils::head(x, n_values))
  converted_head <- convert_array_stream(stream)

  print(converted_head)
  if (more_values > 0) {
    cat(sprintf("...and %d more values\n", more_values))
  }

  invisible(x)
}

#' @export
str.nanoarrow_vctr <- function(object, ...) {
  schema <- attr(object, "schema", exact = TRUE)
  if (is.null(schema)) {
    cat("<nanoarrow_vctr sentinel>\n")
    return(invisible(object))
  }

  formatted <- nanoarrow_schema_formatted(schema, recursive = FALSE)
  cat(sprintf("<nanoarrow_vctr %s[%d]>\n", formatted, length(object)))

  for (chunk in attr(object, "chunks")) {
    str(chunk, ...)
  }

  invisible(object)
}

# Utilities for vctr methods

vctr_resolve_chunk <- function(x, offsets) {
  .Call(nanoarrow_c_vctr_chunk_resolve, x, offsets)
}

vctr_as_slice <- function(x) {
  .Call(nanoarrow_c_vctr_as_slice, x)
}

vctr_array_slice <- function(x, offset, length) {
  new_offset <- x$offset + offset
  new_length <- length
  nanoarrow_array_modify(
    x,
    list(offset = new_offset, length = new_length),
    validate = FALSE
  )
}
