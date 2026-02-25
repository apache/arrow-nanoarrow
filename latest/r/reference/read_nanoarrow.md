# Read/write serialized streams of Arrow data

Reads/writes connections, file paths, URLs, or raw vectors from/to
serialized Arrow data. Arrow documentation typically refers to this
format as "Arrow IPC", since its origin was as a means to transmit
tables between processes (e.g., multiple R sessions). This format can
also be written to and read from files or URLs and is essentially a high
performance equivalent of a CSV file that does a better job maintaining
types.

## Usage

``` r
read_nanoarrow(x, ..., lazy = FALSE)

write_nanoarrow(data, x, ...)
```

## Arguments

- x:

  A [`raw()`](https://rdrr.io/r/base/raw.html) vector, connection, or
  file path from which to read binary data. Common extensions indicating
  compression (.gz, .bz2, .zip) are automatically uncompressed.

- ...:

  Currently unused.

- lazy:

  By default, `read_nanoarrow()` will read and discard a copy of the
  reader's schema to ensure that invalid streams are discovered as soon
  as possible. Use `lazy = TRUE` to defer this check until the reader is
  actually consumed.

- data:

  An object to write as an Arrow IPC stream, converted using
  [`as_nanoarrow_array_stream()`](as_nanoarrow_array_stream.md).
  Notably, this includes a
  [`data.frame()`](https://rdrr.io/r/base/data.frame.html).

## Value

A [nanoarrow_array_stream](as_nanoarrow_array_stream.md)

## Details

The nanoarrow package implements an IPC writer; however, you can also
use
[`arrow::write_ipc_stream()`](https://arrow.apache.org/docs/r/reference/write_ipc_stream.html)
to write data from R, or use the equivalent writer from another Arrow
implementation in Python, C++, Rust, JavaScript, Julia, C#, and beyond.

The media type of an Arrow stream is
`application/vnd.apache.arrow.stream` and the recommended file extension
is `.arrows`.

## Examples

``` r
as.data.frame(read_nanoarrow(example_ipc_stream()))
#>   some_col
#> 1        0
#> 2        1
#> 3        2
```
