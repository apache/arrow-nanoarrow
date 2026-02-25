# Convert an Array Stream into an R vector

Converts `array_stream` to the type specified by `to`. This is a
low-level interface; most users should use
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) or
[`as.vector()`](https://rdrr.io/r/base/vector.html) unless finer-grained
control is needed over the conversion. See
[`convert_array()`](convert_array.md) for details of the conversion
process; see [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md) for
default inferences of `to`.

## Usage

``` r
convert_array_stream(array_stream, to = NULL, size = NULL, n = Inf)

collect_array_stream(array_stream, n = Inf, schema = NULL, validate = TRUE)
```

## Arguments

- array_stream:

  A [nanoarrow_array_stream](as_nanoarrow_array_stream.md).

- to:

  A target prototype object describing the type to which `array` should
  be converted, or `NULL` to use the default conversion as returned by
  [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md). Alternatively,
  a function can be passed to perform an alternative calculation of the
  default ptype as a function of `array` and the default inference of
  the prototype.

- size:

  The exact size of the output, if known. If specified, slightly more
  efficient implementation may be used to collect the output.

- n:

  The maximum number of batches to pull from the array stream.

- schema:

  A [nanoarrow_schema](as_nanoarrow_schema.md) or `NULL` to guess based
  on the first schema.

- validate:

  Use `FALSE` to skip the validation step (i.e., if you know that the
  arrays are valid).

## Value

- `convert_array_stream()`: An R vector of type `to`.

- `collect_array_stream()`: A
  [`list()`](https://rdrr.io/r/base/list.html) of
  [nanoarrow_array](as_nanoarrow_array.md)

## Examples

``` r
stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
str(convert_array_stream(stream))
#> 'data.frame':    5 obs. of  1 variable:
#>  $ x: int  1 2 3 4 5
str(convert_array_stream(stream, to = data.frame(x = double())))
#> 'data.frame':    0 obs. of  1 variable:
#>  $ x: num 

stream <- as_nanoarrow_array_stream(data.frame(x = 1:5))
collect_array_stream(stream)
#> [[1]]
#> <nanoarrow_array struct[5]>
#>  $ length    : int 5
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 1
#>   ..$ :<nanoarrow_buffer validity<bool>[null] ``
#>  $ children  :List of 1
#>   ..$ x:<nanoarrow_array int32[5]>
#>   .. ..$ length    : int 5
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer validity<bool>[null] ``
#>   .. .. ..$ :<nanoarrow_buffer data<int32>[5][20 b]> `1 2 3 4 5`
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
#> 
```
