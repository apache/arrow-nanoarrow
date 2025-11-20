# Convert an Array into an R vector

Converts `array` to the type specified by `to`. This is a low-level
interface; most users should use
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) or
[`as.vector()`](https://rdrr.io/r/base/vector.html) unless finer-grained
control is needed over the conversion. This function is an S3 generic
dispatching on `to`: developers may implement their own S3 methods for
custom vector types.

## Usage

``` r
convert_array(array, to = NULL, ...)
```

## Arguments

- array:

  A [nanoarrow_array](as_nanoarrow_array.md).

- to:

  A target prototype object describing the type to which `array` should
  be converted, or `NULL` to use the default conversion as returned by
  [`infer_nanoarrow_ptype()`](infer_nanoarrow_ptype.md). Alternatively,
  a function can be passed to perform an alternative calculation of the
  default ptype as a function of `array` and the default inference of
  the prototype.

- ...:

  Passed to S3 methods

## Value

An R vector of type `to`.

## Details

Note that unregistered extension types will by default issue a warning.
Use `options(nanoarrow.warn_unregistered_extension = FALSE)` to disable
this behaviour.

Conversions are implemented for the following R vector types:

- [`logical()`](https://rdrr.io/r/base/logical.html): Any numeric type
  can be converted to [`logical()`](https://rdrr.io/r/base/logical.html)
  in addition to the bool type. For numeric types, any non-zero value is
  considered `TRUE`.

- [`integer()`](https://rdrr.io/r/base/integer.html): Any numeric type
  can be converted to
  [`integer()`](https://rdrr.io/r/base/integer.html); however, a warning
  will be signaled if the any value is outside the range of the 32-bit
  integer.

- [`double()`](https://rdrr.io/r/base/double.html): Any numeric type can
  be converted to [`double()`](https://rdrr.io/r/base/double.html). This
  conversion currently does not warn for values that may not roundtrip
  through a floating-point double (e.g., very large uint64 and int64
  values).

- [`character()`](https://rdrr.io/r/base/character.html): String and
  large string types can be converted to
  [`character()`](https://rdrr.io/r/base/character.html). The conversion
  does not check for valid UTF-8: if you need finer-grained control over
  encodings, use `to = blob::blob()`.

- [`factor()`](https://rdrr.io/r/base/factor.html): Dictionary-encoded
  arrays of strings can be converted to
  [`factor()`](https://rdrr.io/r/base/factor.html); however, this must
  be specified explicitly (i.e., `convert_array(array, factor())`)
  because arrays arriving in chunks can have dictionaries that contain
  different levels. Use `convert_array(array, factor(levels = c(...)))`
  to materialize an array into a vector with known levels.

- [Date](https://rdrr.io/r/base/as.Date.html): Only the date32 type can
  be converted to an R Date vector.

- [`hms::hms()`](https://hms.tidyverse.org/reference/hms.html): Time32
  and time64 types can be converted to
  [`hms::hms()`](https://hms.tidyverse.org/reference/hms.html).

- [`difftime()`](https://rdrr.io/r/base/difftime.html): Time32, time64,
  and duration types can be converted to R
  [`difftime()`](https://rdrr.io/r/base/difftime.html) vectors. The
  value is converted to match the
  [`units()`](https://rdrr.io/r/base/units.html) attribute of `to`.

- [`blob::blob()`](https://blob.tidyverse.org/reference/blob.html):
  String, large string, binary, and large binary types can be converted
  to [`blob::blob()`](https://blob.tidyverse.org/reference/blob.html).

- [`vctrs::list_of()`](https://vctrs.r-lib.org/reference/list_of.html):
  List, large list, and fixed-size list types can be converted to
  [`vctrs::list_of()`](https://vctrs.r-lib.org/reference/list_of.html).

- [`matrix()`](https://rdrr.io/r/base/matrix.html): Fixed-size list
  types can be converted to `matrix(ptype, ncol = fixed_size)`.

- [`data.frame()`](https://rdrr.io/r/base/data.frame.html): Struct types
  can be converted to
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html).

- [`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/unspecified.html):
  Any type can be converted to
  [`vctrs::unspecified()`](https://vctrs.r-lib.org/reference/unspecified.html);
  however, a warning will be raised if any non-null values are
  encountered.

In addition to the above conversions, a null array may be converted to
any target prototype except
[`data.frame()`](https://rdrr.io/r/base/data.frame.html). Extension
arrays are currently converted as their storage type.

## Examples

``` r
array <- as_nanoarrow_array(data.frame(x = 1:5))
str(convert_array(array))
#> 'data.frame':    5 obs. of  1 variable:
#>  $ x: int  1 2 3 4 5
str(convert_array(array, to = data.frame(x = double())))
#> 'data.frame':    5 obs. of  1 variable:
#>  $ x: num  1 2 3 4 5
```
