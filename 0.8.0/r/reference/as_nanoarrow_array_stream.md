# Convert an object to a nanoarrow array_stream

In nanoarrow, an 'array stream' corresponds to the
`struct ArrowArrayStream` as defined in the Arrow C Stream interface.
This object is used to represent a stream of
[arrays](as_nanoarrow_array.md) with a common
[schema](as_nanoarrow_schema.md). This is similar to an
[arrow::RecordBatchReader](https://arrow.apache.org/docs/r/reference/RecordBatchReader.html)
except it can be used to represent a stream of any type (not just record
batches). Note that a stream of record batches and a stream of
non-nullable struct arrays are represented identically. Also note that
array streams are mutable objects and are passed by reference and not by
value.

## Usage

``` r
as_nanoarrow_array_stream(x, ..., schema = NULL)
```

## Arguments

- x:

  An object to convert to a array_stream

- ...:

  Passed to S3 methods

- schema:

  An optional schema used to enforce conversion to a particular type.
  Defaults to [`infer_nanoarrow_schema()`](as_nanoarrow_schema.md).

## Value

An object of class 'nanoarrow_array_stream'

## Examples

``` r
(stream <- as_nanoarrow_array_stream(data.frame(x = 1:5)))
#> <nanoarrow_array_stream struct<x: int32>>
#>  $ get_schema:function ()  
#>  $ get_next  :function (schema = x$get_schema(), validate = TRUE)  
#>  $ release   :function ()  
stream$get_schema()
#> <nanoarrow_schema struct>
#>  $ format    : chr "+s"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 0
#>  $ children  :List of 1
#>   ..$ x:<nanoarrow_schema int32>
#>   .. ..$ format    : chr "i"
#>   .. ..$ name      : chr "x"
#>   .. ..$ metadata  : list()
#>   .. ..$ flags     : int 2
#>   .. ..$ children  : list()
#>   .. ..$ dictionary: NULL
#>  $ dictionary: NULL
stream$get_next()
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

# The last batch is returned as NULL
stream$get_next()
#> NULL

# Release the stream
stream$release()
```
