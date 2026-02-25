# Experimental Arrow encoded arrays as R vectors

This experimental vctr class allows zero or more Arrow arrays to present
as an R vector without converting them. This is useful for arrays with
types that do not have a non-lossy R equivalent, and helps provide an
intermediary object type where the default conversion is prohibitively
expensive (e.g., a nested list of data frames). These objects will not
survive many vctr transformations; however, they can be sliced without
copying the underlying arrays.

## Usage

``` r
as_nanoarrow_vctr(x, ..., schema = NULL, subclass = character())

nanoarrow_vctr(schema = NULL, subclass = character())
```

## Arguments

- x:

  An object that works with
  [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md).

- ...:

  Passed to
  [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md)

- schema:

  An optional `schema`

- subclass:

  An optional subclass of nanoarrow_vctr to prepend to the final class
  name.

## Value

A vctr of class 'nanoarrow_vctr'

## Details

The nanoarrow_vctr is currently implemented similarly to
[`factor()`](https://rdrr.io/r/base/factor.html): its storage type is an
[`integer()`](https://rdrr.io/r/base/integer.html) that is a sequence
along the total length of the vctr and there are attributes that are
required to resolve these indices to an array + offset. Sequences
typically have a very compact representation in recent versions of R
such that this has a cheap storage footprint even for large arrays. The
attributes are currently:

- `schema`: The [nanoarrow_schema](as_nanoarrow_schema.md) shared by
  each chunk.

- `chunks`: A [`list()`](https://rdrr.io/r/base/list.html) of
  `nanoarrow_array`.

- `offsets`: An [`integer()`](https://rdrr.io/r/base/integer.html)
  vector beginning with `0` and followed by the cumulative length of
  each chunk. This allows the chunk index + offset to be resolved from a
  logical index with `log(n)` complexity.

This implementation is preliminary and may change; however, the result
of `as_nanoarrow_array_stream(some_vctr[begin:end])` should remain
stable.

## Examples

``` r
array <- as_nanoarrow_array(1:5)
as_nanoarrow_vctr(array)
#> <nanoarrow_vctr int32[5]>
#> [1] 1 2 3 4 5
```
