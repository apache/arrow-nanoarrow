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
#' @param ptr,ptr_src,ptr_dst An external pointer to a `struct ArrowSchema`,
#'   `struct ArrowArray`, or `struct ArrowArrayStream`.
#' @export
#'
nanoarrow_pointer_is_valid <- function(ptr) {
  .Call(nanoarrow_c_pointer_is_valid, ptr)
}

#' @rdname nanoarrow_pointer_is_valid
#' @export
nanoarrow_pointer_addr_chr <- function(ptr) {
  .Call(nanoarrow_c_pointer_addr_chr, ptr)
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
