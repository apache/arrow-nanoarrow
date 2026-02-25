# Create Arrow extension arrays

Create Arrow extension arrays

## Usage

``` r
nanoarrow_extension_array(
  storage_array,
  extension_name,
  extension_metadata = NULL
)
```

## Arguments

- storage_array:

  A [nanoarrow_array](as_nanoarrow_array.md).

- extension_name:

  For [`na_extension()`](na_type.md), the extension name. This is
  typically namespaced separated by dots (e.g., nanoarrow.r.vctrs).

- extension_metadata:

  A string or raw vector defining extension metadata. Most Arrow
  extension types define extension metadata as a JSON object.

## Value

A [nanoarrow_array](as_nanoarrow_array.md) with attached extension
schema.

## Examples

``` r
nanoarrow_extension_array(1:10, "some_ext", '{"key": "value"}')
#> <nanoarrow_array some_ext{int32}[10]>
#>  $ length    : int 10
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 2
#>   ..$ :<nanoarrow_buffer validity<bool>[null] ``
#>   ..$ :<nanoarrow_buffer data<int32>[10][40 b]> `1 2 3 4 5 6 7 8 9 10`
#>  $ dictionary: NULL
#>  $ children  : list()
```
