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

#' Create type objects
#'
#' In nanoarrow, types, fields, and schemas are all represented by a
#' [nanoarrow_schema][as_nanoarrow_schema]. These functions are convenience
#' constructors to create these objects in a readable way. Use [na_type()] to
#' construct types based on the constructor name, which is also the name that
#' prints/is returned by [nanoarrow_schema_parse()].
#'
#' @param type_name The name of the type (e.g., "int32"). This form of the
#'   constructor is useful for writing tests that loop over many types.
#' @param byte_width For [na_fixed_size_binary()], the number of bytes
#'   occupied by each item.
#' @param list_size The number of elements in each item in a
#'   [na_fixed_size_list()].
#' @param precision The total number of digits representable by the decimal type
#' @param scale The number of digits after the decimal point in a decimal type
#' @param unit One of 's' (seconds), 'ms' (milliseconds), 'us' (microseconds),
#'   or 'ns' (nanoseconds).
#' @param timezone A string representing a timezone name. The empty string ""
#'   represents a naive point in time (i.e., one that has no associated
#'   timezone).
#' @param column_types A `list()` of [nanoarrow_schema][as_nanoarrow_schema]s.
#' @param item_type For [na_list()], [na_large_list()], [na_fixed_size_list()],
#'   and [na_map()], the [nanoarrow_schema][as_nanoarrow_schema] representing
#'   the item type.
#' @param key_type The [nanoarrow_schema][as_nanoarrow_schema] representing the
#'   [na_map()] key type.
#' @param index_type The [nanoarrow_schema][as_nanoarrow_schema] representing the
#'   [na_dictionary()] index type.
#' @param value_type The [nanoarrow_schema][as_nanoarrow_schema] representing the
#'   [na_dictionary()] or [na_map()] value type.
#' @param keys_sorted Use `TRUE` to assert that keys are sorted.
#' @param storage_type For [na_extension()], the underlying value type.
#' @param extension_name For [na_extension()], the extension name. This is
#'   typically namespaced separated by dots (e.g., arrow.r.vctrs).
#' @param extension_metadata A string or raw vector defining extension metadata.
#'   Most Arrow extension types define extension metadata as a JSON object.
#' @param nullable Use `FALSE` to assert that this field cannot contain
#'   null values.
#' @param ordered Use `TRUE` to assert that the order of values in the
#'   dictionary are meaningful.
#'
#' @return A [nanoarrow_schema][as_nanoarrow_schema]
#' @export
#'
#' @examples
#' na_int32()
#' na_struct(list(col1 = na_int32()))
#'
na_type <- function(type_name, byte_width = NULL, unit = NULL, timezone = NULL,
                    column_types = NULL, item_type = NULL, key_type = NULL,
                    value_type = NULL, index_type = NULL, ordered = NULL,
                    list_size = NULL, keys_sorted = NULL, storage_type = NULL,
                    extension_name = NULL, extension_metadata = NULL,
                    nullable = NULL) {
  # Create a call and evaluate it. This leads to reasonable error messages
  # regarding nonexistent type names and extraneous or missing parameters.
  args <- list(
    byte_width = byte_width,
    unit = unit,
    timezone = timezone,
    column_types = column_types,
    item_type = item_type,
    key_type = key_type,
    value_type = value_type,
    index_type = index_type,
    ordered = ordered,
    list_size = list_size,
    keys_sorted = keys_sorted,
    storage_type = storage_type,
    extension_name = extension_name,
    extension_metadata = extension_metadata,
    nullable = nullable
  )
  args <- args[!vapply(args, is.null, logical(1))]

  constructor <- as.symbol(paste0("na_", type_name))
  call_obj <- as.call(c(list(constructor), args))
  eval(call_obj)
}

#' @rdname na_type
#' @export
na_na <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE[["NA"]], isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_bool <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$BOOL, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_int8 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INT8, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_uint8 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$UINT8, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_int16 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INT16, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_uint16 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$UINT16, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_int32 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INT32, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_uint32 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$UINT32, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_int64 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INT64, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_uint64 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$UINT64, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_half_float <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$HALF_FLOAT, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_float <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$FLOAT, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_double <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$DOUBLE, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_string <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$STRING, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_large_string <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LARGE_STRING, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_string_view <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$STRING_VIEW, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_binary <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$BINARY, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_large_binary <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LARGE_BINARY, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_fixed_size_binary <- function(byte_width, nullable = TRUE) {
  .Call(
    nanoarrow_c_schema_init_fixed_size,
    NANOARROW_TYPE$FIXED_SIZE_BINARY,
    as.integer(byte_width)[1],
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_binary_view <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$BINARY_VIEW, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_date32 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$DATE32, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_date64 <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$DATE64, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_time32 <- function(unit = c("ms", "s"), nullable = TRUE) {
  unit <- match.arg(unit)
  .Call(
    nanoarrow_c_schema_init_date_time,
    NANOARROW_TYPE$TIME32,
    time_unit_id(unit),
    NULL,
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_time64 <- function(unit = c("us", "ns"), nullable = TRUE) {
  unit <- match.arg(unit)
  .Call(
    nanoarrow_c_schema_init_date_time,
    NANOARROW_TYPE$TIME64,
    time_unit_id(unit),
    NULL,
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_duration <- function(unit = c("ms", "s", "us", "ns"), nullable = TRUE) {
  unit <- match.arg(unit)
  .Call(
    nanoarrow_c_schema_init_date_time,
    NANOARROW_TYPE$DURATION,
    time_unit_id(unit),
    NULL,
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_interval_months <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INTERVAL_MONTHS, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_interval_day_time <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INTERVAL_DAY_TIME, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_interval_month_day_nano <- function(nullable = TRUE) {
  .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$INTERVAL_MONTH_DAY_NANO, isTRUE(nullable))
}

#' @rdname na_type
#' @export
na_timestamp <- function(unit = c("us", "ns", "s", "ms"), timezone = "", nullable = TRUE) {
  unit <- match.arg(unit)
  if (!is.character(timezone) || length(timezone) != 1 || is.na(timezone)) {
    stop("`timezone` must be character(1)")
  }

  .Call(
    nanoarrow_c_schema_init_date_time,
    NANOARROW_TYPE$TIMESTAMP,
    time_unit_id(unit),
    timezone,
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_decimal32 <- function(precision, scale, nullable = TRUE) {
  .Call(
    nanoarrow_c_schema_init_decimal,
    NANOARROW_TYPE$DECIMAL32,
    as.integer(precision)[1],
    as.integer(scale)[1],
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_decimal64 <- function(precision, scale, nullable = TRUE) {
  .Call(
    nanoarrow_c_schema_init_decimal,
    NANOARROW_TYPE$DECIMAL64,
    as.integer(precision)[1],
    as.integer(scale)[1],
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_decimal128 <- function(precision, scale, nullable = TRUE) {
  .Call(
    nanoarrow_c_schema_init_decimal,
    NANOARROW_TYPE$DECIMAL128,
    as.integer(precision)[1],
    as.integer(scale)[1],
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_decimal256 <- function(precision, scale, nullable = TRUE) {
  .Call(
    nanoarrow_c_schema_init_decimal,
    NANOARROW_TYPE$DECIMAL256,
    as.integer(precision)[1],
    as.integer(scale)[1],
    isTRUE(nullable)
  )
}

#' @rdname na_type
#' @export
na_struct <- function(column_types = list(), nullable = FALSE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$STRUCT, isTRUE(nullable))
  schema$children <- column_types
  schema
}

#' @rdname na_type
#' @export
na_sparse_union <- function(column_types = list()) {
  schema <- na_struct(column_types)
  schema$format <- paste0("+us:", paste(seq_along(schema$children) - 1L, collapse = ","))
  schema
}

#' @rdname na_type
#' @export
na_dense_union <- function(column_types = list()) {
  schema <- na_struct(column_types)
  schema$format <- paste0("+ud:", paste(seq_along(schema$children) - 1L, collapse = ","))
  schema
}

#' @rdname na_type
#' @export
na_list <- function(item_type, nullable = TRUE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LIST, isTRUE(nullable))
  schema$children[[1]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_large_list <- function(item_type, nullable = TRUE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LARGE_LIST, isTRUE(nullable))
  schema$children[[1]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_list_view <- function(item_type, nullable = TRUE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LIST_VIEW, isTRUE(nullable))
  schema$children[[1]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_large_list_view <- function(item_type, nullable = TRUE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$LARGE_LIST_VIEW, isTRUE(nullable))
  schema$children[[1]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_fixed_size_list <- function(item_type, list_size, nullable = TRUE) {
  schema <- .Call(
    nanoarrow_c_schema_init_fixed_size,
    NANOARROW_TYPE$FIXED_SIZE_LIST,
    as.integer(list_size)[1],
    isTRUE(nullable)
  )
  schema$children[[1]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_map <- function(key_type, item_type, keys_sorted = FALSE, nullable = TRUE) {
  schema <- .Call(nanoarrow_c_schema_init, NANOARROW_TYPE$MAP, isTRUE(nullable))
  schema$children[[1]]$children[[1]] <- key_type
  schema$children[[1]]$children[[2]] <- item_type
  schema
}

#' @rdname na_type
#' @export
na_dictionary <- function(value_type, index_type = na_int32(), ordered = FALSE) {
  index_type <- as_nanoarrow_schema(index_type)
  index_type$dictionary <- value_type

  if (ordered) {
    index_type$flags <- bitwOr(index_type$flags, ARROW_FLAG$DICTIONARY_ORDERED)
  } else {
    index_type$flags <- bitwAnd(
      index_type$flags,
      bitwNot(ARROW_FLAG$DICTIONARY_ORDERED)
    )
  }

  index_type
}

#' @rdname na_type
#' @export
na_extension <- function(storage_type, extension_name, extension_metadata = "") {
  storage_type <- as_nanoarrow_schema(storage_type)
  new_metadata <- list(
    "ARROW:extension:name" = extension_name,
    "ARROW:extension:metadata" = extension_metadata
  )

  new_metadata <- c(new_metadata, storage_type$metadata)
  storage_type$metadata <- new_metadata[unique(names(new_metadata))]

  storage_type
}

time_unit_id <- function(time_unit) {
  match(time_unit, c("s", "ms", "us", "ns")) - 1L
}

# These values aren't guaranteed to stay stable between nanoarrow versions,
# so we keep them internal but use them in these functions to simplify the
# number of C functions we need to build all the types.
NANOARROW_TYPE <- list(
  UNINITIALIZED = 0,
  "NA" = 1L,
  BOOL = 2L,
  UINT8 = 3L,
  INT8 = 4L,
  UINT16 = 5L,
  INT16 = 6L,
  UINT32 = 7L,
  INT32 = 8L,
  UINT64 = 9L,
  INT64 = 10L,
  HALF_FLOAT = 11L,
  FLOAT = 12L,
  DOUBLE = 13L,
  STRING = 14L,
  BINARY = 15L,
  FIXED_SIZE_BINARY = 16L,
  DATE32 = 17L,
  DATE64 = 18L,
  TIMESTAMP = 19L,
  TIME32 = 20L,
  TIME64 = 21L,
  INTERVAL_MONTHS = 22L,
  INTERVAL_DAY_TIME = 23L,
  DECIMAL128 = 24L,
  DECIMAL256 = 25L,
  LIST = 26L,
  STRUCT = 27L,
  SPARSE_UNION = 28L,
  DENSE_UNION = 29L,
  DICTIONARY = 30L,
  MAP = 31L,
  EXTENSION = 32L,
  FIXED_SIZE_LIST = 33L,
  DURATION = 34L,
  LARGE_STRING = 35L,
  LARGE_BINARY = 36L,
  LARGE_LIST = 37L,
  INTERVAL_MONTH_DAY_NANO = 38L,
  RUN_END_ENCODED = 39L,
  BINARY_VIEW = 40L,
  STRING_VIEW = 41L,
  DECIMAL32 = 42L,
  DECIMAL64 = 43L,
  LIST_VIEW = 44L,
  LARGE_LIST_VIEW = 45L
)

ARROW_FLAG <- list(
  DICTIONARY_ORDERED = 1L,
  NULLABLE = 2L,
  MAP_KEYS_SORTED = 4L
)
