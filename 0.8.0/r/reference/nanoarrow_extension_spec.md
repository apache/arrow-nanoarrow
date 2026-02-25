# Register Arrow extension types

Register Arrow extension types

## Usage

``` r
nanoarrow_extension_spec(data = list(), subclass = character())

register_nanoarrow_extension(extension_name, extension_spec)

unregister_nanoarrow_extension(extension_name)

resolve_nanoarrow_extension(extension_name)
```

## Arguments

- data:

  Optional data to include in the extension type specification

- subclass:

  A subclass for the extension type specification. Extension methods
  will dispatch on this object.

- extension_name:

  An Arrow extension type name (e.g., nanoarrow.r.vctrs)

- extension_spec:

  An extension specification inheriting from 'nanoarrow_extension_spec'.

## Value

- `nanoarrow_extension_spec()` returns an object of class
  'nanoarrow_extension_spec'.

- `register_nanoarrow_extension()` returns `extension_spec`, invisibly.

- `unregister_nanoarrow_extension()` returns `extension_name`,
  invisibly.

- `resolve_nanoarrow_extension()` returns an object of class
  'nanoarrow_extension_spec' or NULL if the extension type was not
  registered.

## Examples

``` r
nanoarrow_extension_spec("mynamespace.mytype", subclass = "mypackage_mytype_spec")
#> [1] "mynamespace.mytype"
#> attr(,"class")
#> [1] "mypackage_mytype_spec"    "nanoarrow_extension_spec"
```
