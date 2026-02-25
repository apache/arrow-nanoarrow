# Create ArrayStreams from batches

Create ArrayStreams from batches

## Usage

``` r
basic_array_stream(batches, schema = NULL, validate = TRUE)
```

## Arguments

- batches:

  A [`list()`](https://rdrr.io/r/base/list.html) of
  [nanoarrow_array](as_nanoarrow_array.md) objects or objects that can
  be coerced via [`as_nanoarrow_array()`](as_nanoarrow_array.md).

- schema:

  A [nanoarrow_schema](as_nanoarrow_schema.md) or `NULL` to guess based
  on the first schema.

- validate:

  Use `FALSE` to skip the validation step (i.e., if you know that the
  arrays are valid).

## Value

An [nanoarrow_array_stream](as_nanoarrow_array_stream.md)

## Examples

``` r
(stream <- basic_array_stream(list(data.frame(a = 1, b = 2))))
#> <nanoarrow_array_stream struct<a: double, b: double>>
#>  $ get_schema:function ()  
#>  $ get_next  :function (schema = x$get_schema(), validate = TRUE)  
#>  $ release   :function ()  
as.data.frame(stream$get_next())
#>   a b
#> 1 1 2
stream$get_next()
#> NULL
```
