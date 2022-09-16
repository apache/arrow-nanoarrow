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

#' Danger zone: low-level pointer operations
#'
#' The [nanoarrow_schema][as_nanoarrow_schema],
#' [nanoarrow_array][as_nanoarrow_array],
#' and [nanoarrow_array_stream][as_nanoarrow_array_stream] classes are
#' represented in R as external pointers (`EXTPTRSXP`). When these objects
#' go out of scope (i.e., when they are garbage collected or shortly
#' thereafter), the underlying object's `release()` callback is called if
#' the underlying pointer is non-null and if the `release()` callback is
#' non-null.
#'
#' When interacting with other C Data Interface implementations, it is
#' important to keep in mind that the R object wrapping these pointers is
#' always passed by reference (because it is an external pointer) and may
#' be referred to by another R object (e.g., an element in a `list()` or as a
#' variable assigned in a user's environment). When importing a schema,
#' array, or array stream into nanoarrow this is not a problem: the R object
#' takes ownership of the lifecycle and memory is released when the R
#' object is garbage collected. In this case, one can use
#' [nanoarrow_pointer_move()] where `ptr_dst` was created using
#' `nanoarrow_allocate_*()`.
#'
#' The case of exporting is more complicated and as such has a dedicated
#' function, [nanoarrow_pointer_export()], that implements different logic
#' schemas, arrays, and array streams:
#'
#' - Schema objects are (deep) copied such that a fresh copy of the schema
#'   is exported and made the responsibility of some other C data interface
#'   implementation.
#' - Array objects are exported as a shell around the original array that
#'   preserves a reference to the R object. This ensures that the buffers
#'   and children pointed to by the array are not copied and that any references
#'   to the original array are not invalidated.
#' - Array stream objects are moved: the responsibility for the object is
#'   transferred to the other C data interface implementation and any
#'   references to the original R object are invalidated. Because these
#'   objects are mutable, this is typically what you want (i.e., you should
#'   not be pulling arrays from a stream accidentally from two places).
#'
#' If you know the lifecycle of your object (i.e., you created the R object
#' yourself and never passed references to it elsewhere), you can slightly
#' more efficiently call [nanoarrow_pointer_move()] for all three pointer
#' types.
#'
#' @param ptr,ptr_src,ptr_dst An external pointer to a `struct ArrowSchema`,
#'   `struct ArrowArray`, or `struct ArrowArrayStream`.
#' @export
#'
nanoarrow_pointer_is_valid <- function(ptr) {
  .Call(nanoarrow_c_pointer_is_valid, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_addr_dbl <- function(ptr) {
  .Call(nanoarrow_c_pointer_addr_dbl, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_addr_chr <- function(ptr) {
  .Call(nanoarrow_c_pointer_addr_chr, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_addr_pretty <- function(ptr) {
  .Call(nanoarrow_c_pointer_addr_pretty, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_release <- function(ptr) {
  invisible(.Call(nanoarrow_c_pointer_release, ptr))
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_move <- function(ptr_src, ptr_dst) {
  invisible(.Call(nanoarrow_c_pointer_move, ptr_src, ptr_dst))
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_export <- function(ptr_src, ptr_dst) {
  if (inherits(ptr_src, "nanoarrow_schema")) {
    invisible(.Call(nanoarrow_c_export_schema, ptr_src, ptr_dst))
  } else if (inherits(ptr_src, "nanoarrow_array")) {
    invisible(.Call(nanoarrow_c_export_array, ptr_src, ptr_dst))
  } else if (inherits(ptr_src, "nanoarrow_array_stream")) {
    # for streams, we don't keep the original pointer alive
    nanoarrow_pointer_move(ptr_src, ptr_dst)
  } else {
    stop(
      "`ptr_src` must inherit from 'nanoarrow_schema', 'nanoarrow_array', or 'nanoarrow_array_stream'"
    )
  }
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_schema <- function() {
  .Call(nanoarrow_c_allocate_schema)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_array <- function() {
  .Call(nanoarrow_c_allocate_array)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_allocate_array_stream <- function() {
  .Call(nanoarrow_c_allocate_array_stream)
}
