# Vctrs extension type

The Arrow format provides a rich type system that can handle most R
vector types; however, many R vector types do not roundtrip perfectly
through Arrow memory. The vctrs extension type uses
[`vctrs::vec_data()`](https://vctrs.r-lib.org/reference/vec_data.html),
[`vctrs::vec_restore()`](https://vctrs.r-lib.org/reference/vec_proxy.html),
and
[`vctrs::vec_ptype()`](https://vctrs.r-lib.org/reference/vec_ptype.html)
in calls to [`as_nanoarrow_array()`](as_nanoarrow_array.md) and
[`convert_array()`](convert_array.md) to ensure roundtrip fidelity.

## Usage

``` r
na_vctrs(ptype, storage_type = NULL)
```

## Arguments

- ptype:

  A vctrs prototype as returned by
  [`vctrs::vec_ptype()`](https://vctrs.r-lib.org/reference/vec_ptype.html).
  The prototype can be of arbitrary size, but a zero-size vector is
  sufficient here.

- storage_type:

  For [`na_extension()`](na_type.md), the underlying value type.

## Value

A [nanoarrow_schema](as_nanoarrow_schema.md).

## Examples

``` r
vctr <- as.POSIXlt("2000-01-02 03:45", tz = "UTC")
array <- as_nanoarrow_array(vctr, schema = na_vctrs(vctr))
infer_nanoarrow_ptype(array)
#> POSIXlt of length 0
convert_array(array)
#> [1] "2000-01-02 03:45:00 UTC"
```
