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


#' Materialize an Array into an R vector
#'
#' @param array A [nanoarrow_array][as_nanoarrow_array].
#' @param to A target prototype object describing the type to which `array`
#'   should be converted, or `NULL` to use the default conversion.
#' @param ... Passed to S3 methods
#'
#' @return An R vector of type `to`.
#' @export
#'
materialize_array <- function(array, to = NULL, ...) {
  stopifnot(inherits(array, "nanoarrow_array"))
  UseMethod("materialize_array", to)
}

#' @export
materialize_array.default <- function(array, to = NULL, ..., .from_c = FALSE) {
  if (.from_c) {
    stop_cant_materialize_array(array, to)
  }

  .Call(nanoarrow_c_materialize_array, array, to)
}

# This is defined because it's verbose to pass named arguments from C.
# When converting data frame columns, we try the internal C conversions
# first to save R evaluation overhead. When the internal conversions fail,
# we call materialize_array() to dispatch to conversions defined via S3
# dispatch, making sure to let the default method know that we've already
# tried the internal C conversions.
materialize_array_from_c <- function(array, to) {
  materialize_array(array, to, .from_c = TRUE)
}

#' @export
materialize_array.vctrs_partial_frame <- function(array, to, ...) {
  ptype <- infer_nanoarrow_ptype(array)
  if (!is.data.frame(ptype)) {
    stop_cant_materialize_array(array, to)
  }

  ptype <- vctrs::vec_ptype_common(ptype, to)
  .Call(nanoarrow_c_materialize_array, array, ptype)
}

#' @export
materialize_array.tbl_df <- function(array, to, ...) {
  df <- materialize_array(array, as.data.frame(to))
  tibble::as_tibble(df)
}

#' @rdname materialize_array
#' @export
infer_nanoarrow_ptype <- function(array) {
  stopifnot(inherits(array, "nanoarrow_array"))
  .Call(nanoarrow_c_infer_ptype, array)
}

# This is called from C from nanoarrow_c_infer_ptype when all the C conversions
# have been tried. Some of these inferences could be moved to C to be faster
# (but are much less verbose to create here)
infer_ptype_other <- function(array) {
  # we don't need the user-friendly versions and this is performance-sensitive
  schema <- .Call(nanoarrow_c_infer_schema_array, array)
  parsed <- .Call(nanoarrow_c_schema_parse, schema)

  switch(
    parsed$type,
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
      ptype <- infer_nanoarrow_ptype(array$children[[1]])
      vctrs::list_of(.ptype = ptype)
    },
    "fixed_size_list" = {
      ptype <- infer_nanoarrow_ptype(array$children[[1]])
      matrix(ptype, nrow = 0, ncol = parsed$fixed_size)
    },
    stop_cant_infer_ptype(array, schema)
  )
}

stop_cant_infer_ptype <- function(array, schema = infer_nanoarrow_schema(array)) {
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

stop_cant_materialize_array <- function(array, to) {
  schema <- infer_nanoarrow_schema(array)
  schema_label <- nanoarrow_schema_formatted(schema)

  if (is.null(schema$name) || identical(schema$name, "")) {
    cnd <- simpleError(
      sprintf(
        "Can't convert array <%s> to R vector of type %s",
        schema_label,
        class(to)[1]
      ),
      call = sys.call(-1)
    )
  } else {
    cnd <- simpleError(
      sprintf(
        "Can't convert `%s` <%s> to R vector of type %s",
        schema$name,
        schema_label,
        class(to)[1]
      ),
      call = sys.call(-1)
    )
  }

  stop(cnd)
}
