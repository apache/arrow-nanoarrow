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

# exported in zzz.R
infer_type.nanoarrow_array <- function(x, ...) {
  arrow::as_data_type(infer_nanoarrow_schema(x, ...))
}

as_data_type.nanoarrow_schema <- function(x, ...) {
  exportable_schema <- nanoarrow_allocate_schema()
  nanoarrow_pointer_export(x, exportable_schema)
  getFromNamespace("DataType", "arrow")$import_from_c(exportable_schema)
}

as_schema.nanoarrow_schema <- function(x, ...) {
  exportable_schema <- nanoarrow_allocate_schema()
  nanoarrow_pointer_export(x, exportable_schema)
  arrow::Schema$import_from_c(exportable_schema)
}

as_arrow_array.nanoarrow_array <- function(x, ..., type = NULL) {
  exportable_schema <- nanoarrow_allocate_schema()
  exportable_array <- nanoarrow_allocate_array()

  schema <- .Call(nanoarrow_c_infer_schema_array, x)
  nanoarrow_pointer_export(schema, exportable_schema)
  nanoarrow_pointer_export(x, exportable_array)

  result <- arrow::Array$import_from_c(exportable_array, exportable_schema)

  if (!is.null(type)) {
    result$cast(type)
  } else {
    result
  }
}

as_chunked_array.nanoarrow_array <- function(x, ..., type = NULL) {
  arrow::as_chunked_array(as_arrow_array.nanoarrow_array(x, ..., type = type))
}

as_record_batch.nanoarrow_array <- function(x, ..., schema = NULL) {
  exportable_schema <- nanoarrow_allocate_schema()
  exportable_array <- nanoarrow_allocate_array()

  nanoarrow_pointer_export(
    .Call(nanoarrow_c_infer_schema_array, x),
    exportable_schema
  )
  nanoarrow_pointer_export(x, exportable_array)

  result <- arrow::RecordBatch$import_from_c(exportable_array, exportable_schema)

  if (!is.null(schema)) {
    arrow::as_record_batch(result, schema = schema)
  } else {
    result
  }
}

as_arrow_table.nanoarrow_array <- function(x, ..., schema = NULL) {
  arrow::as_arrow_table(
    as_record_batch.nanoarrow_array(x, schema = schema)
  )
}

as_record_batch_reader.nanoarrow_array_stream <- function(x, ..., schema = NULL) {
  # TODO: not supporting an explicit schema here yet
  stopifnot(is.null(schema))

  # RecordBatchReaders are mutable and shouldn't be pulled from more than one
  # place, so it's safe to move them and invalidate any R references to `x`
  arrow::RecordBatchReader$import_from_c(x)
}

#' @export
as_nanoarrow_schema.DataType <- function(x, ...) {
  schema <- nanoarrow_allocate_schema()
  x$export_to_c(schema)
  schema
}

#' @export
as_nanoarrow_schema.Field <- function(x, ...) {
  schema <- nanoarrow_allocate_schema()
  x$export_to_c(schema)
  schema
}

#' @export
as_nanoarrow_schema.Schema <- function(x, ...) {
  schema <- nanoarrow_allocate_schema()
  x$export_to_c(schema)
  schema
}

#' @export
as_nanoarrow_array.Array <- function(x, ..., schema = NULL) {
  imported_schema <- nanoarrow_allocate_schema()
  array <- nanoarrow_allocate_array()

  if (!is.null(schema)) {
    x <- x$cast(arrow::as_data_type(schema))
  }

  x$export_to_c(array, imported_schema)

  nanoarrow_array_set_schema(array, imported_schema)
  array
}

#' @export
as_nanoarrow_array.ChunkedArray <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    array <- arrow::as_arrow_array(x)
  } else {
    array <- arrow::as_arrow_array(x, type = arrow::as_data_type(schema))
  }

  as_nanoarrow_array.Array(array)
}

#' @export
as_nanoarrow_array.RecordBatch <- function(x, ..., schema = NULL) {
  imported_schema <- nanoarrow_allocate_schema()
  array <- nanoarrow_allocate_array()

  if (!is.null(schema)) {
    x <- x$cast(arrow::as_schema(schema))
  }

  x$export_to_c(array, imported_schema)

  nanoarrow_array_set_schema(array, imported_schema)
  array
}

#' @export
as_nanoarrow_array.Table <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    batch <- arrow::as_record_batch(x)
  } else {
    batch <- arrow::as_record_batch(x, schema = arrow::as_schema(schema))
  }

  as_nanoarrow_array.RecordBatch(batch)
}

#' @export
as_nanoarrow_array_stream.RecordBatchReader <- function(x, ..., schema = NULL) {
  # TODO: not supporting an explicit schema here yet
  stopifnot(is.null(schema))
  array_stream <- nanoarrow_allocate_array_stream()
  x$export_to_c(array_stream)
  array_stream
}
