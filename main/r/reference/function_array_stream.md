# Create an array stream from an R function

The function is called once for each requested array. It should return
an object coercible by [`as_nanoarrow_array()`](as_nanoarrow_array.md),
or `NULL` to indicate end of stream. The schema is required because it
is available before the first array is requested. Neither construction
nor schema access calls `fun`. After `NULL` or an error is returned, the
callback is not called again.

## Usage

``` r
function_array_stream(fun, schema, validate = TRUE)
```

## Arguments

- fun:

  A function called with no arguments to produce each array.

- schema:

  A schema coercible by
  [`as_nanoarrow_schema()`](as_nanoarrow_schema.md), used to coerce and
  validate each result.

- validate:

  A single `TRUE` or `FALSE`. Use `FALSE` to skip validation within the
  callback. The stream's `get_next()` method can validate the returned
  array independently.

## Value

An [nanoarrow_array_stream](as_nanoarrow_array_stream.md)

## Details

Errors and interrupts raised by `fun` are converted to stream errors.
Evaluation is also isolated in a top-level R context so other non-local
exits (for example, invoking an `abort` restart) cannot escape through
the C stream interface. Calling handlers in the caller are therefore not
applied to callback evaluation. The callback is evaluated only on R's
main thread. Warnings are handled in this isolated context, unless
handled within `fun`. `fun` must not release the stream (or call its
methods) while it is being evaluated. The stream retains `fun` until it
is released; explicitly release streams when `fun` or its environment
refers to the stream itself.

## Examples

``` r
i <- 0L
stream <- function_array_stream(
  function() {
    i <<- i + 1L
    if (i > 2L) NULL else mtcars[i, , drop = FALSE]
  },
  infer_nanoarrow_schema(mtcars)
)
as.data.frame(stream$get_next())
#>   mpg cyl disp  hp drat   wt  qsec vs am gear carb
#> 1  21   6  160 110  3.9 2.62 16.46  0  1    4    4
as.data.frame(stream$get_next())
#>   mpg cyl disp  hp drat    wt  qsec vs am gear carb
#> 1  21   6  160 110  3.9 2.875 17.02  0  1    4    4
stream$get_next()
#> NULL
stream$release()
```
