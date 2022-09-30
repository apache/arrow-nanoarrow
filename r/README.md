
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

You can install the development version of nanoarrow from
[GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("apache/arrow-nanoarrow/r", build = FALSE)
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
arrow R package `DataType` like `arrow::int32()`).

``` r
infer_nanoarrow_schema(1:5)
#> <nanoarrow_schema int32>
#>  $ format    : chr "i"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 2
#>  $ children  : NULL
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
#>   .. ..$ children  : NULL
#>   .. ..$ dictionary: NULL
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
#>   ..$ :<nanoarrow_buffer_validity[0 b] at 0x0>
#>   ..$ :<nanoarrow_buffer_data_int32[20 b] at 0x135d13c28>
#>  $ dictionary: NULL
#>  $ children  : list()
as_nanoarrow_array(arrow::record_batch(col1 = c(1.1, 2.2)))
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
#>   .. .. ..$ :<nanoarrow_buffer_data_double[16 b] at 0x13604f0b8>
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
```

You can use `as.vector()` or `as.data.frame()` to get the R
representation of the object back:

``` r
array <- as_nanoarrow_array(arrow::record_batch(col1 = c(1.1, 2.2)))
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
#>   .. ..$ children  : NULL
#>   .. ..$ dictionary: NULL
#>  $ dictionary: NULL
```

### Array Streams

The easiest way to create an ArrowArrayStream is from an
`arrow::RecordBatchReader`:

``` r
reader <- arrow::RecordBatchReader$create(
  arrow::record_batch(col1 = c(1.1, 2.2)),
  arrow::record_batch(col1 = c(3.3, 4.4))
)

(stream <- as_nanoarrow_array_stream(reader))
#> <nanoarrow_array_stream struct<col1: double>>
#>  $ get_schema:function ()  
#>  $ get_next  :function (schema = x$get_schema(), validate = TRUE)  
#>  $ release   :function ()
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
#>   .. .. ..$ :<nanoarrow_buffer_data_double[16 b] at 0x136de3538>
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
#>   .. .. ..$ :<nanoarrow_buffer_data_double[16 b] at 0x136de3178>
#>   .. ..$ dictionary: NULL
#>   .. ..$ children  : list()
#>  $ dictionary: NULL
stream$get_next()
#> NULL
```

After consuming a stream, you should call the release method as soon as
you can. This lets the implementation of the stream release any
resources (like open files) it may be holding in a more predictable way
than waiting for the garbage collector to clean up the object.
