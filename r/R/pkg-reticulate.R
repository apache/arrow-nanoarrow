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


#' Python integration via reticulate
#'
#' These functions enable Python wrapper objects created via reticulate to
#' be used with any function that uses [as_nanoarrow_array()] or
#' [as_nanoarrow_array_stream()] to accept generic "arrowable" input.
#' Implementations for [reticulate::py_to_r()] and [reticulate::r_to_py()]
#' are also included such that nanoarrow's array/schema/array stream objects
#' can be passed as arguments to Python functions that would otherwise accept
#' an object implementing the Arrow PyCapsule protocol.
#'
#' This implementation uses the
#' [Arrow PyCapsule protocol](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
#' to interpret an arbitrary Python object as an Arrow array/schema/array stream
#' and produces Python objects that implement this protocol. This is currently
#' implemented using the nanoarrow Python package which provides similar
#' primitives for facilitating interchange in Python.
#'
#' @param x An Python object to convert
#' @param schema A requested schema, which may or may not be honoured depending
#'   on the capabilities of the producer
#' @param ... Unused
#'
#' @returns
#'   - `as_nanoarrow_schema()` returns an object of class nanoarrow_schema
#'   - `as_nanoarrow_array()` returns an object of class nanoarrow_array
#'   - `as_nanoarrow_array_stream()` returns an object of class
#'     nanoarrow_array_stream.
#' @export
#'
#' @examplesIf identical(Sys.getenv("NANOARROW_R_TEST_RETICULATE"), "true")
#' library(reticulate)
#'
#' py_require("nanoarrow")
#'
#' na <- import("nanoarrow", convert = FALSE)
#' python_arrayish_thing <- na$Array(1:3, na_int32())
#' as_nanoarrow_array(python_arrayish_thing)
#'
#' r_to_py(as_nanoarrow_array(1:3))
as_nanoarrow_schema.python.builtin.object <- function(x, ...) {
  na <- reticulate::import("nanoarrow", convert = FALSE)
  c_schema <- na$c_schema(x)

  schema_dst <- nanoarrow_allocate_schema()
  nanoarrow_pointer_move(
    reticulate::py_str(c_schema[["_addr"]]()),
    schema_dst
  )

  schema_dst
}

#' @rdname as_nanoarrow_schema.python.builtin.object
#' @export
as_nanoarrow_array.python.builtin.object <- function(x, ..., schema = NULL) {
  if (!is.null(schema)) {
    schema <- reticulate::r_to_py(as_nanoarrow_schema(schema), convert = FALSE)
  }

  na <- reticulate::import("nanoarrow", convert = FALSE)
  c_array <- na$c_array(x, schema)

  schema_dst <- nanoarrow_allocate_schema()
  array_dst <-  nanoarrow_allocate_array()
  nanoarrow_pointer_move(
    reticulate::py_str(c_array$schema[["_addr"]]()),
    schema_dst
  )
  nanoarrow_pointer_move(
    reticulate::py_str(c_array[["_addr"]]()),
    array_dst
  )

  nanoarrow_array_set_schema(array_dst, schema_dst, validate = FALSE)
  array_dst
}

#' @rdname as_nanoarrow_schema.python.builtin.object
#' @export
as_nanoarrow_array_stream.python.builtin.object <- function(x, ..., schema = NULL) {
  if (!is.null(schema)) {
    schema <- reticulate::r_to_py(as_nanoarrow_schema(schema), convert = FALSE)
  }

  na <- reticulate::import("nanoarrow", convert = FALSE)
  c_array_stream <- na$c_array_stream(x, schema)

  array_stream_dst <- nanoarrow_allocate_array_stream()
  nanoarrow_pointer_move(
    reticulate::py_str(c_array_stream[["_addr"]]()),
    array_stream_dst
  )

  array_stream_dst
}

r_to_py.nanoarrow_schema <- function(x, convert = FALSE) {
  na_c_schema <- reticulate::import("nanoarrow.c_schema", convert = FALSE)

  out <- na_c_schema$allocate_c_schema()
  out_addr <- reticulate::py_str(out[["_addr"]]())
  nanoarrow_pointer_export(x, out_addr)

  na <- reticulate::import("nanoarrow", convert = FALSE)
  na$Schema(out)
}

r_to_py.nanoarrow_array <- function(x, convert = FALSE) {
  na_c_array <- reticulate::import("nanoarrow.c_array", convert = FALSE)

  out <- na_c_array$allocate_c_array()
  out_addr <- reticulate::py_str(out[["_addr"]]())
  out_schema_addr <- reticulate::py_str(out$schema[["_addr"]]())

  nanoarrow_pointer_export(infer_nanoarrow_schema(x), out_schema_addr)
  nanoarrow_pointer_export(x, out_addr)

  na <- reticulate::import("nanoarrow", convert = FALSE)
  na$Array(out)
}

r_to_py.nanoarrow_array_stream <- function(x, convert = FALSE) {
  na_c_array_stream <- reticulate::import("nanoarrow.c_array_stream", convert = FALSE)

  out <- na_c_array_stream$allocate_c_array_stream()
  out_addr <- reticulate::py_str(out[["_addr"]]())
  nanoarrow_pointer_export(x, out_addr)

  na <- reticulate::import("nanoarrow", convert = FALSE)
  na$ArrayStream(out)
}

py_to_r.nanoarrow.schema.Schema <- function(x) {
  as_nanoarrow_schema(x)
}

py_to_r.nanoarrow.array.Array <- function(x) {
  as_nanoarrow_array(x)
}

py_to_r.nanoarrow.array_stream.ArrayStream <- function(x) {
  as_nanoarrow_array_stream(x)
}

#' @rdname as_nanoarrow_schema.python.builtin.object
#' @export
has_reticulate_with_nanoarrow <- function() {
  requireNamespace("reticulate", quietly = TRUE) &&
    reticulate::py_available() &&
    !inherits(try(reticulate::import("nanoarrow"), silent = TRUE), "try-error")
}
