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

#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom utils getFromNamespace
#' @useDynLib nanoarrow, .registration = TRUE
## usethis namespace: end
NULL

#' Underlying 'nanoarrow' C library build
#'
#' @param runtime Compare TRUE and FALSE values to detect a
#'   possible ABI mismatch.
#'
#' @return A string identifying the version of nanoarrow this package
#'   was compiled against.
#' @export
#'
#' @examples
#' nanoarrow_version()
#' nanoarrow_with_zstd()
nanoarrow_version <- function(runtime = TRUE) {
  if (runtime) {
    .Call(nanoarrow_c_version_runtime)
  } else {
    .Call(nanoarrow_c_version)
  }
}

#' @rdname nanoarrow_version
#' @export
nanoarrow_with_zstd <- function() {
  .Call(nanoarrow_c_with_zstd)
}
