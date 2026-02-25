# nanoarrow

The goal of nanoarrow is to provide minimal useful bindings to the
[Arrow C Data](https://arrow.apache.org/docs/format/CDataInterface.html)
and [Arrow C
Stream](https://arrow.apache.org/docs/format/CStreamInterface.html)
interfaces using the [nanoarrow C
library](https://arrow.apache.org/nanoarrow/).

## Installation

You can install the released version of nanoarrow from
[CRAN](https://cran.r-project.org/) with:

``` r
install.packages("nanoarrow")
```

You can install the development version of nanoarrow from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("apache/arrow-nanoarrow/r")
```

If you can load the package, you’re good to go!

``` r
library(nanoarrow)
```

## Example

The Arrow C Data and Arrow C Stream interfaces are comprised of three
structures: the `ArrowSchema` which represents a data type of an array,
the `ArrowArray` which represents the values of an array, and an
`ArrowArrayStream`, which represents zero or more `ArrowArray`s with a
common `ArrowSchema`. All three can be wrapped by R objects using the
nanoarrow R package.

### Schemas

Use [`infer_nanoarrow_schema()`](reference/as_nanoarrow_schema.md) to
get the ArrowSchema object that corresponds to a given R vector type;
use [`as_nanoarrow_schema()`](reference/as_nanoarrow_schema.md) to
convert an object from some other data type representation (e.g., an
arrow R package `DataType` like
[`arrow::int32()`](https://arrow.apache.org/docs/r/reference/data-type.html));
or use `na_XXX()` functions to construct them.

``` r
infer_nanoarrow_schema(1:5)
#> <nanoarrow_schema int32>
#>  $ format    : chr "i"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 2
#>  $ children  : list()
#>  $ dictionary: NULL
as_nanoarrow_schema(arrow::schema(col1 = arrow::float64()))
#> <nanoarrow_schema struct>
#>  $ format    : chr "+s"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 0
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_schema double>
#>   .. ..$ format    : chr "g"
#>   .. ..$ name      : chr "col1"
#>   .. ..$ metadata  : list()
#>   .. ..$ flags     : int 2
#>   .. ..$ children  : list()
#>   .. ..$ dictionary: NULL
#>  $ dictionary: NULL
na_int64()
#> <nanoarrow_schema int64>
#>  $ format    : chr "l"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 2
#>  $ children  : list()
#>  $ dictionary: NULL
```

### Arrays

Use [`as_nanoarrow_array()`](reference/as_nanoarrow_array.md) to convert
an object to an ArrowArray object:

``` r
as_nanoarrow_array(1:5)
#> <nanoarrow_array int32[5]>
#>  $ length    : int 5
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 2
#>   ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>   ..$ :<nanoarrow_buffer data<int32>[5][20 b]> `1 2 3 4 5`
#>  $ dictionary: NULL
#>  $ children  : list()
as_nanoarrow_array(data.frame(col1 = c(1.1, 2.2)))
#> <nanoarrow_array struct[2]>
#>  $ length    : int 2
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 1
#>   ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_array double[2]>
#>   .. ..$ length    : int 2
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>   .. .. ..$ :<nanoarrow_buffer data<double>[2][16 b]> `1.1 2.2`
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
```

You can use [`as.vector()`](https://rdrr.io/r/base/vector.html) or
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) to get
the R representation of the object back:

``` r
array <- as_nanoarrow_array(data.frame(col1 = c(1.1, 2.2)))
as.data.frame(array)
#>   col1
#> 1  1.1
#> 2  2.2
```

Even though at the C level the ArrowArray is distinct from the
ArrowSchema, at the R level we attach a schema wherever possible. You
can access the attached schema using
[`infer_nanoarrow_schema()`](reference/as_nanoarrow_schema.md):

``` r
infer_nanoarrow_schema(array)
#> <nanoarrow_schema struct>
#>  $ format    : chr "+s"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 0
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_schema double>
#>   .. ..$ format    : chr "g"
#>   .. ..$ name      : chr "col1"
#>   .. ..$ metadata  : list()
#>   .. ..$ flags     : int 2
#>   .. ..$ children  : list()
#>   .. ..$ dictionary: NULL
#>  $ dictionary: NULL
```

### Array Streams

The easiest way to create an ArrowArrayStream is from a list of arrays
or objects that can be converted to an array using
[`as_nanoarrow_array()`](reference/as_nanoarrow_array.md):

``` r
stream <- basic_array_stream(
  list(
    data.frame(col1 = c(1.1, 2.2)),
    data.frame(col1 = c(3.3, 4.4))
  )
)
```

You can pull batches from the stream using the `$get_next()` method. The
last batch will return `NULL`.

``` r
stream$get_next()
#> <nanoarrow_array struct[2]>
#>  $ length    : int 2
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 1
#>   ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_array double[2]>
#>   .. ..$ length    : int 2
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>   .. .. ..$ :<nanoarrow_buffer data<double>[2][16 b]> `1.1 2.2`
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
stream$get_next()
#> <nanoarrow_array struct[2]>
#>  $ length    : int 2
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 1
#>   ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_array double[2]>
#>   .. ..$ length    : int 2
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer validity<bool>[0][0 b]> ``
#>   .. .. ..$ :<nanoarrow_buffer data<double>[2][16 b]> `3.3 4.4`
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
stream$get_next()
#> NULL
```

You can pull all the batches into a
[`data.frame()`](https://rdrr.io/r/base/data.frame.html) by calling
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) or
[`as.vector()`](https://rdrr.io/r/base/vector.html):

``` r
stream <- basic_array_stream(
  list(
    data.frame(col1 = c(1.1, 2.2)),
    data.frame(col1 = c(3.3, 4.4))
  )
)

as.data.frame(stream)
#>   col1
#> 1  1.1
#> 2  2.2
#> 3  3.3
#> 4  4.4
```

After consuming a stream, you should call the release method as soon as
you can. This lets the implementation of the stream release any
resources (like open files) it may be holding in a more predictable way
than waiting for the garbage collector to clean up the object.

## Integration with the arrow package

The nanoarrow package implements
[`as_nanoarrow_schema()`](reference/as_nanoarrow_schema.md),
[`as_nanoarrow_array()`](reference/as_nanoarrow_array.md), and
[`as_nanoarrow_array_stream()`](reference/as_nanoarrow_array_stream.md)
for most arrow package types. Similarly, it implements
[`arrow::as_arrow_array()`](https://arrow.apache.org/docs/r/reference/as_arrow_array.html),
[`arrow::as_record_batch()`](https://arrow.apache.org/docs/r/reference/as_record_batch.html),
[`arrow::as_arrow_table()`](https://arrow.apache.org/docs/r/reference/as_arrow_table.html),
[`arrow::as_record_batch_reader()`](https://arrow.apache.org/docs/r/reference/as_record_batch_reader.html),
[`arrow::infer_type()`](https://arrow.apache.org/docs/r/reference/infer_type.html),
[`arrow::as_data_type()`](https://arrow.apache.org/docs/r/reference/as_data_type.html),
and
[`arrow::as_schema()`](https://arrow.apache.org/docs/r/reference/as_schema.html)
for nanoarrow objects such that you can pass equivalent nanoarrow
objects into many arrow functions and vice versa.
