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

#' Infer an R vector prototype
#'
#' Resolves the default `ptype` value to use in [materialize_array()].
#'
#' @param x A [nanoarrow_schema][as_nanoarrow_schema],
#'   [nanoarrow_array][as_nanoarrow_array], or
#'   [nanoarrow_array_stream][as_nanoarrow_array_stream].
#'
#' @return An R vector of zero size describing the target into which
#'   the array should be materialized.
#' @export
#'
#' @examples
#' infer_nanoarrow_ptype(as_nanoarrow_array(1:10))
#'
infer_nanoarrow_ptype <- function(x) {
  if (inherits(x, "nanoarrow_array")) {
    x <- .Call(nanoarrow_c_infer_schema_array, x)
  } else if (inherits(x, "nanoarrow_array_stream")) {
    x <- .Call(nanoarrow_c_array_stream_get_schema, x)
  } else if (!inherits(x, "nanoarrow_schema")) {
    stop("`x` must be a nanoarrow_schema(), nanoarrow_array(), or nanoarrow_array_stream()")
  }

  .Call(nanoarrow_c_infer_ptype, x)
}

# This is called from C from nanoarrow_c_infer_ptype when all the C conversions
# have been tried. Some of these inferences could be moved to C to be faster
# (but are much less verbose to create here)
infer_ptype_other <- function(schema) {
  # we don't need the user-friendly versions and this is performance-sensitive
  parsed <- .Call(nanoarrow_c_schema_parse, schema)

  switch(
    parsed$type,
    "na" = vctrs::unspecified(),
    "binary" = ,
    "large_binary" = blob::new_blob(),
    "date32" = ,
    "date64" = structure(numeric(), class = "Date"),
    "time32" = ,
    "time64" = hms::hms(),
    "duration" = structure(numeric(), class = "difftime", units = "secs"),
    "timestamp" = {
      if (parsed$timezone == "") {
        # We almost never want to assume the user's timezone here, which is
        # what would happen if we passed on "". This is consistent with how
        # readr handles reading timezones (assign "UTC" since it's DST-free
        # and let the user explicitly set this later)
        parsed$timezone <- getOption("nanoarrow.timezone_if_unspecified", "UTC")
      }

      structure(
        numeric(0),
        class = c("POSIXct", "POSIXt"),
        tzone = parsed$timezone
      )
    },
    "large_list" = ,
    "list" = {
      ptype <- infer_nanoarrow_ptype(schema$children[[1]])
      vctrs::list_of(.ptype = ptype)
    },
    "fixed_size_list" = {
      ptype <- infer_nanoarrow_ptype(schema$children[[1]])
      matrix(ptype, nrow = 0, ncol = parsed$fixed_size)
    },
    stop_cant_infer_ptype(schema)
  )
}

stop_cant_infer_ptype <- function(schema) {
  if (is.null(schema$name) || identical(schema$name, "")) {
    cnd <- simpleError(
      sprintf(
        "Can't infer R vector type for array <%s>",
        schema$format
      ),
      call = sys.call(-1)
    )
  } else {
    cnd <- simpleError(
      sprintf(
        "Can't infer R vector type for `%s` <%s>",
        schema$name,
        schema$format
      ),
      call = sys.call(-1)
    )
  }

  stop(cnd)
}
