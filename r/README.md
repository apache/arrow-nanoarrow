
<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->
<!-- README.md is generated from README.Rmd. Please edit that file -->

# nanoarrow

<!-- badges: start -->
<!-- badges: end -->

The goal of nanoarrow is to provide minimal useful bindings to the
[Arrow C Data](https://arrow.apache.org/docs/format/CDataInterface.html)
and [Arrow C
Stream](https://arrow.apache.org/docs/format/CStreamInterface.html)
interfaces using the [nanoarrow C
library](https://apache.github.io/arrow-nanoarrow/).

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

Use `infer_nanoarrow_schema()` to get the ArrowSchema object that
corresponds to a given R vector type; use `as_nanoarrow_schema()` to
convert an object from some other data type representation (e.g., an
arrow R package `DataType` like `arrow::int32()`); or use `na_XXX()`
functions to construct them.

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

Use `as_nanoarrow_array()` to convert an object to an ArrowArray object:

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

You can use `as.vector()` or `as.data.frame()` to get the R
representation of the object back:

``` r
array <- as_nanoarrow_array(data.frame(col1 = c(1.1, 2.2)))
as.data.frame(array)
#>   col1
#> 1  1.1
#> 2  2.2
```

Even though at the C level the ArrowArray is distinct from the
ArrowSchema, at the R level we attach a schema wherever possible. You
can access the attached schema using `infer_nanoarrow_schema()`:

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
`as_nanoarrow_array()`:

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
#>   ..$ :<nanoarrow_buffer_validity[0 b] at 0x0>
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_array double[2]>
#>   .. ..$ length    : int 2
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer_validity[0 b] at 0x0>
#>   .. .. ..$ :<nanoarrow_buffer_data_double[16 b] at 0x13af88b38>
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
stream$get_next()
#> <nanoarrow_array struct[2]>
#>  $ length    : int 2
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 1
#>   ..$ :<nanoarrow_buffer_validity[0 b] at 0x0>
#>  $ children  :List of 1
#>   ..$ col1:<nanoarrow_array double[2]>
#>   .. ..$ length    : int 2
#>   .. ..$ null_count: int 0
#>   .. ..$ offset    : int 0
#>   .. ..$ buffers   :List of 2
#>   .. .. ..$ :<nanoarrow_buffer_validity[0 b] at 0x0>
#>   .. .. ..$ :<nanoarrow_buffer_data_double[16 b] at 0x13af886f8>
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
stream$get_next()
#> NULL
```

You can pull all the batches into a `data.frame()` by calling
`as.data.frame()` or `as.vector()`:

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

The nanoarrow package implements `as_nanoarrow_schema()`,
`as_nanoarrow_array()`, and `as_nanoarrow_array_stream()` for most arrow
package types. Similarly, it implements `arrow::as_arrow_array()`,
`arrow::as_record_batch()`, `arrow::as_arrow_table()`,
`arrow::as_record_batch_reader()`, `arrow::infer_type()`,
`arrow::as_data_type()`, and `arrow::as_schema()` for nanoarrow objects
such that you can pass equivalent nanoarrow objects into many arrow
functions and vice versa.
