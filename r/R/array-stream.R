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

#' Convert an object to a nanoarrow array_stream
#'
#' @param x An object to convert to a array_stream
#' @param ... Passed to S3 methods
#' @inheritParams as_nanoarrow_array
#'
#' @return An object of class 'nanoarrow_array_stream'
#' @export
as_nanoarrow_array_stream <- function(x, ..., schema = NULL) {
  UseMethod("as_nanoarrow_array_stream")
}

#' @export
as_nanoarrow_array_stream.nanoarrow_array_stream <- function(x, ..., schema = NULL) {
  if (is.null(schema)) {
    x
  } else {
    NextMethod()
  }
}

#' @export
as_nanoarrow_array_stream.default <- function(x, ..., schema = NULL) {
  as_nanoarrow_array_stream(
    arrow::as_record_batch_reader(x, ...),
    schema = schema
  )
}
