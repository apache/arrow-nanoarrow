# Underlying 'nanoarrow' C library build

Underlying 'nanoarrow' C library build

## Usage

``` r
nanoarrow_version(runtime = TRUE)

nanoarrow_with_zstd()
```

## Arguments

- runtime:

  Compare TRUE and FALSE values to detect a possible ABI mismatch.

## Value

A string identifying the version of nanoarrow this package was compiled
against.

## Examples

``` r
nanoarrow_version()
#> [1] "0.9.0-SNAPSHOT"
nanoarrow_with_zstd()
#> [1] TRUE
```
