# Modify nanoarrow arrays

Create a new array or from an existing array, modify one or more
parameters. When importing an array from elsewhere,
`nanoarrow_array_set_schema()` is useful to attach the data type
information to the array (without this information there is little that
nanoarrow can do with the array since its content cannot be otherwise
interpreted). `nanoarrow_array_modify()` can create a shallow copy and
modify various parameters to create a new array, including setting
children and buffers recursively. These functions power the `$<-`
operator, which can modify one parameter at a time.

## Usage

``` r
nanoarrow_array_init(schema)

nanoarrow_array_set_schema(array, schema, validate = TRUE)

nanoarrow_array_modify(array, new_values, validate = TRUE)
```

## Arguments

- schema:

  A [nanoarrow_schema](as_nanoarrow_schema.md) to attach to this
  `array`.

- array:

  A [nanoarrow_array](as_nanoarrow_array.md).

- validate:

  Use `FALSE` to skip validation. Skipping validation may result in
  creating an array that will crash R.

- new_values:

  A named [`list()`](https://rdrr.io/r/base/list.html) of values to
  replace.

## Value

- `nanoarrow_array_init()` returns a possibly invalid but initialized
  array with a given `schema`.

- `nanoarrow_array_set_schema()` returns `array`, invisibly. Note that
  `array` is modified in place by reference.

- `nanoarrow_array_modify()` returns a shallow copy of `array` with the
  modified parameters such that the original array remains valid.

## Examples

``` r
nanoarrow_array_init(na_string())
#> <nanoarrow_array string[0]>
#>  $ length    : int 0
#>  $ null_count: int 0
#>  $ offset    : int 0
#>  $ buffers   :List of 3
#>   ..$ :<nanoarrow_buffer validity<bool>[null] ``
#>   ..$ :<nanoarrow_buffer data_offset<int32>[null] ``
#>   ..$ :<nanoarrow_buffer data<string>[null] ``
#>  $ dictionary: NULL
#>  $ children  : list()

# Modify an array using $ and <-
array <- as_nanoarrow_array(1:5)
array$length <- 4
as.vector(array)
#> [1] 1 2 3 4

# Modify potentially more than one component at a time
array <- as_nanoarrow_array(1:5)
as.vector(nanoarrow_array_modify(array, list(length = 4)))
#> [1] 1 2 3 4

# Attach a schema to an array
array <- as_nanoarrow_array(-1L)
nanoarrow_array_set_schema(array, na_uint32())
as.vector(array)
#> [1] 4294967295
```
