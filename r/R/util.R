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

`%||%` <- function(rhs, lhs) {
  if (is.null(rhs)) lhs else rhs
}

new_data_frame <- function(x, nrow) {
  structure(x, row.names = c(NA, nrow), class = "data.frame")
}

vec_gen <- function(ptype, n = 1e3, prop_true = 0.5,  prop_na = 0,
                    chr_len = function(n) ceiling(25 * stats::runif(n))) {
  vec <- switch(
    class(ptype)[1],
    logical = stats::runif(n) < prop_true,
    integer = as.integer(stats::runif(n, min = -1, max = 1) * .Machine$integer.max),
    numeric = stats::runif(n),
    character = strrep(rep_len(letters, n), chr_len(n)),
    data.frame = new_data_frame(
      lapply(
        ptype,
        vec_gen,
        n = n,
        prop_true = prop_true,
        prop_na = prop_na,
        chr_len = chr_len
      ),
      n
    ),
    stop(sprintf("Don't know how to generate vector for type %s", class(ptype)[1]))
  )

  if (!is.data.frame(vec) && prop_na > 0) {
    is_na <- stats::runif(n) < prop_na
    vec[is_na] <- ptype[NA_integer_]
  }

  vec
}

vec_shuffle <- function(x) {
  if (is.data.frame(x)) {
    x[sample(seq_len(nrow(x)), replace = FALSE), , drop = FALSE]
  } else {
    x[sample(seq_along(x), replace = FALSE)]
  }
}
