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
#' This experimental vctr class allows zero or more Arrow arrays to
#' present as an R vector without converting them. This is useful for arrays
#' with types that do not have a non-lossy R equivalent, and helps provide an
#' intermediary object type where the default conversion is prohibitively
#' expensive (e.g., a nested list of data frames). These objects will not
#' survive many vctr transformations; however, they can be sliced without
#' copying the underlying arrays.
#'
#' The nanoarrow_vctr is currently implemented similarly to `factor()`: its
#' storage type is an `integer()` that is a sequence along the total length
#' of the vctr and there are attributes that are required to resolve these
#' indices to an array + offset. Sequences typically have a very compact
#' representation in recent versions of R such that this has a cheap storage
#' footprint even for large arrays. The attributes are currently:
#'
#' - `schema`: The [nanoarrow_schema][as_nanoarrow_schema] shared by each chunk.
#' - `chunks`: A `list()` of `nanoarrow_array`.
#' - `offsets`: An `integer()` vector beginning with `0` and followed by the
#'   cumulative length of each chunk. This allows the chunk index + offset
#'   to be resolved from a logical index with `log(n)` complexity.
#'
#' This implementation is preliminary and may change; however, the result of
#' `as_nanoarrow_array_stream(some_vctr[begin:end])` should remain stable.
#'
#' @param x An object that works with [as_nanoarrow_array_stream()].
#' @param subclass An optional subclass of nanoarrow_vctr to prepend to the
#'   final class name.
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
as_nanoarrow_vctr <- function(x, ..., schema = NULL, subclass = character()) {
  if (inherits(x, "nanoarrow_vctr") && is.null(schema)) {
    return(x)
  }

  stream <- as_nanoarrow_array_stream(x, ..., schema = schema)
  chunks <- collect_array_stream(stream, validate = FALSE)
  new_nanoarrow_vctr(chunks, stream$get_schema(), subclass)
}

#' @rdname as_nanoarrow_vctr
#' @export
nanoarrow_vctr <- function(schema = NULL, subclass = character()) {
  if (is.null(schema)) {
    new_nanoarrow_vctr(list(), NULL, subclass)
  } else {
    new_nanoarrow_vctr(list(), as_nanoarrow_schema(schema), subclass)
  }
}

new_nanoarrow_vctr <- function(chunks, schema, subclass = character()) {
  offsets <- .Call(nanoarrow_c_vctr_chunk_offsets, chunks)
  indices <- seq_len(offsets[length(offsets)])

  structure(
    indices,
    schema = schema,
    chunks = chunks,
    offsets = offsets,
    class = union(subclass, "nanoarrow_vctr")
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
  if (length(x) == 0) {
    return(character())
  }

  stream <- as_nanoarrow_array_stream(x)
  converted <- convert_array_stream(stream)

  # This needs to be a character() with the same length as x to work with
  # RStudio's viewer. Data frames need special handling in this case.
  size_stable_format(converted)
}

size_stable_format <- function(x, ...) {
  if (inherits(x, "nanoarrow_vctr")) {
    # Extension types could have a default convert that gives a nanoarrow_vctr.
    # If this is the case, they should be returning a subclass with a format
    # method that ensures we don't get here.
    rep(sprintf("<%s[%d]>", class(x)[1], seq_along(x)))
  } else if (inherits(x, "data.frame")) {
    cols <- lapply(x, size_stable_format, ...)
    cols <- Map(paste, names(x), cols, sep = ": ")
    rows <- do.call(paste, c(cols, list(sep = ", ")))
    paste0("{", rows, "}")
  } else {
    format(x, ...)
  }
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
    # If a schema is passed, first resolve the stream as is and then use
    # as_nanoarrow_array_stream() to either cast (when this is supported)
    # or error.
    stream <- as_nanoarrow_array_stream(x, schema = NULL)
    return(as_nanoarrow_array_stream(stream, schema = schema))
  }

  # Resolve the indices as c(1-based start, length)
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
  dots <- list(...)

  # This one we can do safely
  if (length(dots) == 1) {
    return(dots[[1]])
  }

  stop("c() not implemented for nanoarrow_vctr()")
}

# Ensures that nanoarrow_vctr can fit in a data.frame
#' @export
as.data.frame.nanoarrow_vctr <- function(x, ..., optional = FALSE) {
  if (!optional) {
    stop(sprintf("cannot coerce object of tyoe '%s' to data.frame", class(x)[1]))
  } else {
    new_data_frame(list(x), nrow = length(x))
  }
}

#' @export
print.nanoarrow_vctr <- function(x, ...) {
  schema <- attr(x, "schema", exact = TRUE)
  if (is.null(schema)) {
    cat(sprintf("<%s <any>>\n", class(x)[1]))
    return(invisible(x))
  }

  formatted <- nanoarrow_schema_formatted(schema, recursive = FALSE)
  cat(sprintf("<%s %s[%d]>\n", class(x)[1], formatted, length(x)))

  n_values <- min(length(x), 20)
  more_values <- length(x) - n_values
  stream <- as_nanoarrow_array_stream(utils::head(x, n_values))
  converted_head <- convert_array_stream(stream)

  if (inherits(converted_head, "nanoarrow_vctr")) {
    converted_head <- format(converted_head)
    print(converted_head, quote = FALSE)
  } else {
    print(converted_head)
  }

  if (more_values >= 2) {
    cat(sprintf("...and %d more values\n", more_values))
  } else if (more_values >= 1) {
    cat(sprintf("...and %d more value\n", more_values))
  }

  invisible(x)
}

#' @export
str.nanoarrow_vctr <- function(object, ...) {
  schema <- attr(object, "schema", exact = TRUE)
  if (is.null(schema)) {
    cat(sprintf("<%s <any>>\n", class(object)[1]))
    return(invisible(object))
  }

  formatted <- nanoarrow_schema_formatted(schema, recursive = FALSE)
  cat(sprintf("<%s %s[%d]>\n", class(object)[1], formatted, length(object)))

  # Prints out the C data interface dump of each chunk with the chunk
  # index above.
  str(attr(object, "chunks"))

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
