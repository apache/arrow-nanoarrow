
#' @export
as_nanoarrow_schema.DataType <- function(x, ...) {
  schema <- nanoarrow_allocate_schema()
  x$export_to_c(schema)
  schema
}

#' @export
as_nanoarrow_array_stream.RecordBatchReader <- function(x, ...) {
  array_stream <- nanoarrow_allocate_array_stream()
  x$export_to_c(array_stream)
  array_stream
}

#' @export
as_nanoarrow_array.Array <- function(x, ...) {
  schema <- nanoarrow_allocate_schema()
  array <- nanoarrow_allocate_array()
  x$export_to_c(array, schema)

  # TODO: haven't sorted how to encode array + schema (probably as the tag)
  array
}
