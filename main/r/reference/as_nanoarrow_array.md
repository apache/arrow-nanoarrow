# Convert an object to a nanoarrow array

In nanoarrow an 'array' refers to the `struct ArrowArray` definition in
the Arrow C data interface. At the R level, we attach a
[schema](as_nanoarrow_schema.md) such that functionally the
nanoarrow_array class can be used in a similar way as an
[`arrow::Array`](https://arrow.apache.org/docs/r/reference/array-class.html).
Note that in nanoarrow an
[`arrow::RecordBatch`](https://arrow.apache.org/docs/r/reference/RecordBatch-class.html)
and a non-nullable
[`arrow::StructArray`](https://arrow.apache.org/docs/r/reference/array-class.html)
are represented identically.

## Usage

``` r
as_nanoarrow_array(x, ..., schema = NULL)
```

## Arguments

- x:

  An object to convert to a array

- ...:

  Passed to S3 methods

- schema:

  An optional schema used to enforce conversion to a particular type.
  Defaults to [`infer_nanoarrow_schema()`](as_nanoarrow_schema.md).

## Value

An object of class 'nanoarrow_array'

## Examples

``` r
(array <- as_nanoarrow_array(1:5))
#> <nanoarrow_array int32[5]>
#>  $ length    : int 5
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 2
#>   ..$ :<nanoarrow_buffer validity<bool>[null] ``
#>   ..$ :<nanoarrow_buffer data<int32>[5][20 b]> `1 2 3 4 5`
#>  $ dictionary: NULL
#>  $ children  : list()
as.vector(array)
#> [1] 1 2 3 4 5

(array <- as_nanoarrow_array(data.frame(x = 1:5)))
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
as.data.frame(array)
#>   x
#> 1 1
#> 2 2
#> 3 3
#> 4 4
#> 5 5
```
