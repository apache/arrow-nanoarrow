# Convert an object to a nanoarrow schema

In nanoarrow a 'schema' refers to a `struct ArrowSchema` as defined in
the Arrow C Data interface. This data structure can be used to represent
an
[`arrow::schema()`](https://arrow.apache.org/docs/r/reference/schema.html),
an
[`arrow::field()`](https://arrow.apache.org/docs/r/reference/Field-class.html),
or an
[`arrow::DataType`](https://arrow.apache.org/docs/r/reference/DataType-class.html).
Note that in nanoarrow, an
[`arrow::schema()`](https://arrow.apache.org/docs/r/reference/schema.html)
and a non-nullable
[`arrow::struct()`](https://arrow.apache.org/docs/r/reference/data-type.html)
are represented identically.

## Usage

``` r
as_nanoarrow_schema(x, ...)

infer_nanoarrow_schema(x, ...)

nanoarrow_schema_parse(x, recursive = FALSE)

nanoarrow_schema_modify(x, new_values, validate = TRUE)
```

## Arguments

- x:

  An object to convert to a schema

- ...:

  Passed to S3 methods

- recursive:

  Use `TRUE` to include a `children` member when parsing schemas.

- new_values:

  New schema component to assign

- validate:

  Use `FALSE` to skip schema validation

## Value

An object of class 'nanoarrow_schema'

## Examples

``` r
infer_nanoarrow_schema(integer())
#> <nanoarrow_schema int32>
#>  $ format    : chr "i"
#>  $ name      : chr ""
#>  $ metadata  : list()
#>  $ flags     : int 2
#>  $ children  : list()
#>  $ dictionary: NULL
infer_nanoarrow_schema(data.frame(x = integer()))
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
```
