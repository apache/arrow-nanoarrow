# Create type objects

In nanoarrow, types, fields, and schemas are all represented by a
[nanoarrow_schema](as_nanoarrow_schema.md). These functions are
convenience constructors to create these objects in a readable way. Use
`na_type()` to construct types based on the constructor name, which is
also the name that prints/is returned by
[`nanoarrow_schema_parse()`](as_nanoarrow_schema.md).

## Usage

``` r
na_type(
  type_name,
  byte_width = NULL,
  unit = NULL,
  timezone = NULL,
  precision = NULL,
  scale = NULL,
  column_types = NULL,
  item_type = NULL,
  key_type = NULL,
  value_type = NULL,
  index_type = NULL,
  ordered = NULL,
  list_size = NULL,
  keys_sorted = NULL,
  storage_type = NULL,
  extension_name = NULL,
  extension_metadata = NULL,
  nullable = NULL
)

na_na(nullable = TRUE)

na_bool(nullable = TRUE)

na_int8(nullable = TRUE)

na_uint8(nullable = TRUE)

na_int16(nullable = TRUE)

na_uint16(nullable = TRUE)

na_int32(nullable = TRUE)

na_uint32(nullable = TRUE)

na_int64(nullable = TRUE)

na_uint64(nullable = TRUE)

na_half_float(nullable = TRUE)

na_float(nullable = TRUE)

na_double(nullable = TRUE)

na_string(nullable = TRUE)

na_large_string(nullable = TRUE)

na_string_view(nullable = TRUE)

na_binary(nullable = TRUE)

na_large_binary(nullable = TRUE)

na_fixed_size_binary(byte_width, nullable = TRUE)

na_binary_view(nullable = TRUE)

na_date32(nullable = TRUE)

na_date64(nullable = TRUE)

na_time32(unit = c("ms", "s"), nullable = TRUE)

na_time64(unit = c("us", "ns"), nullable = TRUE)

na_duration(unit = c("ms", "s", "us", "ns"), nullable = TRUE)

na_interval_months(nullable = TRUE)

na_interval_day_time(nullable = TRUE)

na_interval_month_day_nano(nullable = TRUE)

na_timestamp(unit = c("us", "ns", "s", "ms"), timezone = "", nullable = TRUE)

na_decimal32(precision, scale, nullable = TRUE)

na_decimal64(precision, scale, nullable = TRUE)

na_decimal128(precision, scale, nullable = TRUE)

na_decimal256(precision, scale, nullable = TRUE)

na_struct(column_types = list(), nullable = FALSE)

na_sparse_union(column_types = list())

na_dense_union(column_types = list())

na_list(item_type, nullable = TRUE)

na_large_list(item_type, nullable = TRUE)

na_list_view(item_type, nullable = TRUE)

na_large_list_view(item_type, nullable = TRUE)

na_fixed_size_list(item_type, list_size, nullable = TRUE)

na_map(key_type, item_type, keys_sorted = FALSE, nullable = TRUE)

na_dictionary(value_type, index_type = na_int32(), ordered = FALSE)

na_extension(storage_type, extension_name, extension_metadata = "")
```

## Arguments

- type_name:

  The name of the type (e.g., "int32"). This form of the constructor is
  useful for writing tests that loop over many types.

- byte_width:

  For `na_fixed_size_binary()`, the number of bytes occupied by each
  item.

- unit:

  One of 's' (seconds), 'ms' (milliseconds), 'us' (microseconds), or
  'ns' (nanoseconds).

- timezone:

  A string representing a timezone name. The empty string "" represents
  a naive point in time (i.e., one that has no associated timezone).

- precision:

  The total number of digits representable by the decimal type

- scale:

  The number of digits after the decimal point in a decimal type

- column_types:

  A [`list()`](https://rdrr.io/r/base/list.html) of
  [nanoarrow_schema](as_nanoarrow_schema.md)s.

- item_type:

  For `na_list()`, `na_large_list()`, `na_fixed_size_list()`, and
  `na_map()`, the [nanoarrow_schema](as_nanoarrow_schema.md)
  representing the item type.

- key_type:

  The [nanoarrow_schema](as_nanoarrow_schema.md) representing the
  `na_map()` key type.

- value_type:

  The [nanoarrow_schema](as_nanoarrow_schema.md) representing the
  `na_dictionary()` or `na_map()` value type.

- index_type:

  The [nanoarrow_schema](as_nanoarrow_schema.md) representing the
  `na_dictionary()` index type.

- ordered:

  Use `TRUE` to assert that the order of values in the dictionary are
  meaningful.

- list_size:

  The number of elements in each item in a `na_fixed_size_list()`.

- keys_sorted:

  Use `TRUE` to assert that keys are sorted.

- storage_type:

  For `na_extension()`, the underlying value type.

- extension_name:

  For `na_extension()`, the extension name. This is typically namespaced
  separated by dots (e.g., nanoarrow.r.vctrs).

- extension_metadata:

  A string or raw vector defining extension metadata. Most Arrow
  extension types define extension metadata as a JSON object.

- nullable:

  Use `FALSE` to assert that this field cannot contain null values.

## Value

A [nanoarrow_schema](as_nanoarrow_schema.md)

## Examples

``` r
na_int32()
#> <nanoarrow_schema int32>
#>  $ format    : chr "i"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 2
#>  $ children  : list()
#>  $ dictionary: NULL
na_struct(list(col1 = na_int32()))
#> <nanoarrow_schema struct>
#>  $ format    : chr "+s"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 0
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_schema int32>
#>   .. ..$ format    : chr "i"
#>   .. ..$ name      : chr "col1"
#>   .. ..$ metadata  : list()
#>   .. ..$ flags     : int 2
#>   .. ..$ children  : list()
#>   .. ..$ dictionary: NULL
#>  $ dictionary: NULL
```
