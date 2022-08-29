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

#' Convert an object to a nanoarrow array
#'
#' @param x An object to convert to a array
#' @param schema An optional schema used to enforce conversion to a particular
#'   type. Defaults to [infer_nanoarrow_schema()].
#' @param ... Passed to S3 methods
#'
#' @return An object of class 'nanoarrow_array'
#' @export
as_nanoarrow_array <- function(x, ..., schema = infer_nanoarrow_schema(x)) {
  UseMethod("as_nanoarrow_array")
}

#' @export
as_nanoarrow_array.default <- function(x, ..., schema = infer_nanoarrow_schema(x)) {
  as_nanoarrow_array(arrow::as_arrow_array(x, type = arrow::as_data_type(schema)))
}
