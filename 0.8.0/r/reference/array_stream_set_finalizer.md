# Register an array stream finalizer

In some cases, R functions that return a
[nanoarrow_array_stream](as_nanoarrow_array_stream.md) may require that
the scope of some other object outlive that of the array stream. If
there is a need for that object to be released deterministically (e.g.,
to close open files), you can register a function to run after the
stream's release callback is invoked from the R thread. Note that this
finalizer will **not** be run if the stream's release callback is
invoked from a **non**-R thread. In this case, the finalizer and its
chain of environments will be garbage-collected when
`nanoarrow:::preserved_empty()` is run.

## Usage

``` r
array_stream_set_finalizer(array_stream, finalizer)
```

## Arguments

- array_stream:

  A [nanoarrow_array_stream](as_nanoarrow_array_stream.md)

- finalizer:

  A function that will be called with zero arguments.

## Value

A newly allocated `array_stream` whose release callback will call the
supplied finalizer.

## Examples

``` r
stream <- array_stream_set_finalizer(
  basic_array_stream(list(1:5)),
  function() message("All done!")
)
stream$release()
#> All done!
```
