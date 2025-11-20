# Python integration via reticulate

These functions enable Python wrapper objects created via reticulate to
be used with any function that uses
[`as_nanoarrow_array()`](as_nanoarrow_array.md) or
[`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md) to accept
generic "arrowable" input. Implementations for
[`reticulate::py_to_r()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.html)
and
[`reticulate::r_to_py()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.html)
are also included such that nanoarrow's array/schema/array stream
objects can be passed as arguments to Python functions that would
otherwise accept an object implementing the Arrow PyCapsule protocol.

## Usage

``` r
# S3 method for class 'python.builtin.object'
as_nanoarrow_schema(x, ...)

# S3 method for class 'python.builtin.object'
as_nanoarrow_array(x, ..., schema = NULL)

# S3 method for class 'python.builtin.object'
as_nanoarrow_array_stream(x, ..., schema = NULL)

test_reticulate_with_nanoarrow()
```

## Arguments

- x:

  An Python object to convert

- ...:

  Unused

- schema:

  A requested schema, which may or may not be honoured depending on the
  capabilities of the producer

## Value

- [`as_nanoarrow_schema()`](as_nanoarrow_schema.md) returns an object of
  class nanoarrow_schema

- [`as_nanoarrow_array()`](as_nanoarrow_array.md) returns an object of
  class nanoarrow_array

- [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md) returns
  an object of class nanoarrow_array_stream.

## Details

This implementation uses the [Arrow PyCapsule
protocol](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
to interpret an arbitrary Python object as an Arrow array/schema/array
stream and produces Python objects that implement this protocol. This is
currently implemented using the nanoarrow Python package which provides
similar primitives for facilitating interchange in Python.

## Examples

``` r
if (FALSE) { # test_reticulate_with_nanoarrow()
library(reticulate)

py_require("nanoarrow")

na <- import("nanoarrow", convert = FALSE)
python_arrayish_thing <- na$Array(1:3, na_int32())
as_nanoarrow_array(python_arrayish_thing)

r_to_py(as_nanoarrow_array(1:3))
}
```
