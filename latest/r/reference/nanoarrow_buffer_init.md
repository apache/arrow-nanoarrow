# Create and modify nanoarrow buffers

Create and modify nanoarrow buffers

## Usage

``` r
nanoarrow_buffer_init()

nanoarrow_buffer_append(buffer, new_buffer)

convert_buffer(buffer, to = NULL)
```

## Arguments

- buffer, new_buffer:

  [nanoarrow_buffer](as_nanoarrow_buffer.md)s.

- to:

  A target prototype object describing the type to which `array` should
  be converted, or `NULL` to use the default conversion as returned by
  [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md). Alternatively,
  a function can be passed to perform an alternative calculation of the
  default ptype as a function of `array` and the default inference of
  the prototype.

## Value

- `nanoarrow_buffer_init()`: An object of class 'nanoarrow_buffer'

- `nanoarrow_buffer_append()`: Returns `buffer`, invisibly. Note that
  `buffer` is modified in place by reference.

## Examples

``` r
buffer <- nanoarrow_buffer_init()
nanoarrow_buffer_append(buffer, 1:5)

array <- nanoarrow_array_modify(
  nanoarrow_array_init(na_int32()),
  list(length = 5, buffers = list(NULL, buffer))
)
as.vector(array)
#> [1] 1 2 3 4 5
```
