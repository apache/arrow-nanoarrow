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

# For testing the altrep chr conversion
nanoarrow_altrep_chr <- function(array) {
  .Call(nanoarrow_c_make_altrep_chr, array)
}

is_nanoarrow_altrep <- function(x) {
  .Call(nanoarrow_c_is_altrep, x)
}

nanoarrow_altrep_force_materialize <- function(x, recursive = FALSE) {
  invisible(.Call(nanoarrow_c_altrep_force_materialize, x, recursive))
}

is_nanoarrow_altrep_materialized <- function(x) {
  .Call(nanoarrow_c_altrep_is_materialized, x)
}
