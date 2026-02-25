# Implement Arrow extension types

Implement Arrow extension types

## Usage

``` r
infer_nanoarrow_ptype_extension(
  extension_spec,
  x,
  ...,
  warn_unregistered = TRUE
)

convert_array_extension(
  extension_spec,
  array,
  to,
  ...,
  warn_unregistered = TRUE
)

as_nanoarrow_array_extension(extension_spec, x, ..., schema = NULL)
```

## Arguments

- extension_spec:

  An extension specification inheriting from 'nanoarrow_extension_spec'.

- x, array, to, schema, ...:

  Passed from [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md),
  [`convert_array()`](convert_array.md),
  [`as_nanoarrow_array()`](as_nanoarrow_array.md), and/or
  [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md).

- warn_unregistered:

  Use `FALSE` to infer/convert based on the storage type without a
  warning.

## Value

- `infer_nanoarrow_ptype_extension()`: The R vector prototype to be used
  as the default conversion target.

- `convert_array_extension()`: An R vector of type `to`.

- `as_nanoarrow_array_extension()`: A
  [nanoarrow_array](as_nanoarrow_array.md) of type `schema`.
