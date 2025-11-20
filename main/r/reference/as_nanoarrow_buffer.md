# Convert an object to a nanoarrow buffer

Convert an object to a nanoarrow buffer

## Usage

``` r
as_nanoarrow_buffer(x, ...)
```

## Arguments

- x:

  An object to convert to a buffer

- ...:

  Passed to S3 methods

## Value

An object of class 'nanoarrow_buffer'

## Examples

``` r
array <- as_nanoarrow_array(c(NA, 1:4))
array$buffers
#> [[1]]
#> <nanoarrow_buffer validity<bool>[8][1 b]> `FALSE TRUE TRUE TRUE TRUE FALSE F...`
#> 
#> [[2]]
#> <nanoarrow_buffer data<int32>[5][20 b]> `NA 1 2 3 4`
#> 
as.raw(array$buffers[[1]])
#> [1] 1e
as.raw(array$buffers[[2]])
#>  [1] 00 00 00 80 01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00
convert_buffer(array$buffers[[1]])
#> [1] FALSE  TRUE  TRUE  TRUE  TRUE FALSE FALSE FALSE
convert_buffer(array$buffers[[2]])
#> [1] NA  1  2  3  4
```
