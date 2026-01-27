# Infer an R vector prototype

Resolves the default `to` value to use in
[`convert_array()`](convert_array.md) and
[`convert_array_stream()`](convert_array_stream.md). The default
conversions are:

## Usage

``` r
infer_nanoarrow_ptype(x)
```

## Arguments

- x:

  A [nanoarrow_schema](as_nanoarrow_schema.md),
  [nanoarrow_array](as_nanoarrow_array.md), or
  [nanoarrow_array_stream](as_nanoarrow_array_stream.md).

## Value

An R vector of zero size describing the target into which the array
should be materialized.

## Details

- null to
  [`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/vctrs-unspecified.html)

- boolean to [`logical()`](https://rdrr.io/r/base/logical.html)

- int8, uint8, int16, uint16, and int13 to
  [`integer()`](https://rdrr.io/r/base/integer.html)

- uint32, int64, uint64, float, and double to
  [`double()`](https://rdrr.io/r/base/double.html)

- string and large string to
  [`character()`](https://rdrr.io/r/base/character.html)

- struct to [`data.frame()`](https://rdrr.io/r/base/data.frame.html)

- binary and large binary to
  [`blob::blob()`](https://blob.tidyverse.org/reference/blob.html)

- list, large_list, and fixed_size_list to
  [`vctrs::list_of()`](https://vctrs.r-lib.org/reference/list_of.html)

- time32 and time64 to
  [`hms::hms()`](https://hms.tidyverse.org/reference/hms.html)

- duration to [`difftime()`](https://rdrr.io/r/base/difftime.html)

- date32 to [`as.Date()`](https://rdrr.io/r/base/as.Date.html)

- timestamp to [`as.POSIXct()`](https://rdrr.io/r/base/as.POSIXlt.html)

Additional conversions are possible by specifying an explicit value for
`to`. For details of each conversion, see
[`convert_array()`](convert_array.md).

## Examples

``` r
infer_nanoarrow_ptype(as_nanoarrow_array(1:10))
#> integer(0)
```
