
#' Create type objects
#'
#' In nanoarow, types, fields, and schemas are all represented by a
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
na_type <- function(type_name, byte_width = NULL, unit = NULL, timezone = NULL,
                    column_types = NULL, item_type = NULL, key_type = NULL,
                    value_type = NULL, index_type = NULL, ordered = NULL,
                    list_size = NULL, keys_sorted = NULL, storage_type = NULL,
                    extension_name = NULL, extension_metadata = NULL,
                    nullable = NULL) {
  # Create a call and evaluate it. This leads to reasonable error messages
  # regarding nonexistent type names and extraneous or missing parameters.
  constructor <- paste0("na_", type_name)
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

  call_obj <- as.call(c(list(constructor), args))
  eval(call_obj)
}

#' @rdname na_type
#' @export
na_na <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_bool <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_int8 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_uint8 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_int16 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_uint16 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_int32 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_uint32 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_int64 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_uint64 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_half_float <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_float <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_double <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_string <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_large_string <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_binary <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_large_binary <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_fixed_size_binary <- function(byte_width, nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_date32 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_date64 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_time32 <- function(unit = c("ms", "s"), nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_time64 <- function(unit = c("ns", "us"), nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_duration <- function(unit = c("s", "ms", "us", "ns"), nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_timestamp <- function(unit = c("s", "ms", "us", "ns"), timezone = "", nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_decimal128 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_decimal256 <- function(nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_struct <- function(column_types = list(), nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_list <- function(item_type, nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_large_list <- function(item_type, nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_fixed_size_list <- function(item_type, list_size, nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_map <- function(key_type, item_type, keys_sorted = FALSE, nullable = TRUE) {

}

#' @rdname na_type
#' @export
na_dictionary <- function(value_type, index_type = na_int32(), ordered = FALSE) {

}

#' @rdname na_type
#' @export
na_extension <- function(storage_type, extension_name, extension_metadata = "") {

}
