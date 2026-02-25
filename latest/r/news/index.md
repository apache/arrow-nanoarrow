# Changelog

## nanoarrow 0.8.0

- Add reticulate/Python integration
  ([\#817](https://github.com/apache/arrow-nanoarrow/issues/817))
- Add support for creating timestamp and duration types from numeric
  storage
  ([\#816](https://github.com/apache/arrow-nanoarrow/issues/816))
- Correct invocation of R_MakeExternalPtr with R NULL
  ([\#841](https://github.com/apache/arrow-nanoarrow/issues/841))
- Fix usage of deperecated syntax for forthcoming R release
  ([\#840](https://github.com/apache/arrow-nanoarrow/issues/840))
- Fix map type and struct-nested-in-struct convert
  ([\#829](https://github.com/apache/arrow-nanoarrow/issues/829))
- Collect array streams in C (not R) before conversion
  ([\#828](https://github.com/apache/arrow-nanoarrow/issues/828))
- Fix test for forthcoming ALTREP behaviour in R-devel
  ([\#826](https://github.com/apache/arrow-nanoarrow/issues/826),
  [\#842](https://github.com/apache/arrow-nanoarrow/issues/842))
- Ensure C23 version check works for clang16 (current GitHub Actions)
  ([\#801](https://github.com/apache/arrow-nanoarrow/issues/801))

## nanoarrow 0.7.0

CRAN release: 2025-07-03

- Add zstd decompression support to R package
  ([\#733](https://github.com/apache/arrow-nanoarrow/issues/733))
- Support native creation of more numeric Arrow arrays from integer
  vectors
  ([\#697](https://github.com/apache/arrow-nanoarrow/issues/697))
- Support matrix objects as fixed-size-list arrays
  ([\#692](https://github.com/apache/arrow-nanoarrow/issues/692))
- Ensure that `python` is used on Windows when running bootstrap.R
  ([\#792](https://github.com/apache/arrow-nanoarrow/issues/792))
- Update vctrs extension name to reflect implementation change
  ([\#752](https://github.com/apache/arrow-nanoarrow/issues/752))
- sub-day precision Date should be floored when treated as integer
  ([\#674](https://github.com/apache/arrow-nanoarrow/issues/674))

## nanoarrow 0.6.0

CRAN release: 2024-10-13

- Add float16 support for R bindings
  ([\#650](https://github.com/apache/arrow-nanoarrow/issues/650))
- Implement string view support in R bindings
  ([\#636](https://github.com/apache/arrow-nanoarrow/issues/636))
- Allow opt-out of warning for unregistered extension types
  ([\#632](https://github.com/apache/arrow-nanoarrow/issues/632))
- Add bindings to IPC writer
  ([\#608](https://github.com/apache/arrow-nanoarrow/issues/608))
- Avoid flatcc aligned_alloc() call when compiling R package
  ([\#494](https://github.com/apache/arrow-nanoarrow/issues/494))
- Use JSON in experimental R vctrs extension type
  ([\#533](https://github.com/apache/arrow-nanoarrow/issues/533))

## nanoarrow 0.5.0

CRAN release: 2024-05-26

- Add experimental `nanoarrow_vctr` to wrap a list of arrays
  ([\#461](https://github.com/apache/arrow-nanoarrow/issues/461))
- Add bindings for IPC reader
  ([\#390](https://github.com/apache/arrow-nanoarrow/issues/390))
- Fix tests for platforms where arrow dataset and/or zip is not
  available
  ([\#415](https://github.com/apache/arrow-nanoarrow/issues/415))
- Remove unnecessary package name call
  ([\#381](https://github.com/apache/arrow-nanoarrow/issues/381))

## nanoarrow 0.4.0

CRAN release: 2024-02-01

- Fix source links from pkgdown site
  ([\#315](https://github.com/apache/arrow-nanoarrow/issues/315)).
- Provide LinkingTo headers for extension packages
  ([\#332](https://github.com/apache/arrow-nanoarrow/issues/332)).
- Add more `nanoarrow_array_stream` generics
  ([\#349](https://github.com/apache/arrow-nanoarrow/issues/349)).
- Add conversion from integer type to
  [`character()`](https://rdrr.io/r/base/character.html)
  ([\#345](https://github.com/apache/arrow-nanoarrow/issues/345)).
- Ensure simple [`list()`](https://rdrr.io/r/base/list.html)s can be
  converted without arrow installed
  ([\#344](https://github.com/apache/arrow-nanoarrow/issues/344)).

## nanoarrow 0.3.0.1

CRAN release: 2023-12-08

- Ensure wrapper array stream eagerly releases the wrapped array stream
  ([\#333](https://github.com/apache/arrow-nanoarrow/issues/333)).

## nanoarrow 0.3.0

CRAN release: 2023-09-29

- Use classed warnings to signal that a lossy conversion occurred
  ([\#298](https://github.com/apache/arrow-nanoarrow/issues/298))
- Add support for
  [`bit64::integer64()`](https://rdrr.io/pkg/bit64/man/bit64-package.html)
  conversions
  ([\#293](https://github.com/apache/arrow-nanoarrow/issues/293))
- Implement extension type registration/conversion
  ([\#288](https://github.com/apache/arrow-nanoarrow/issues/288))
- Implement dictionary conversion
  ([\#285](https://github.com/apache/arrow-nanoarrow/issues/285))
- Ensure `ordered` is reflected in
  [`na_dictionary()`](../reference/na_type.md)
  ([\#299](https://github.com/apache/arrow-nanoarrow/issues/299))
- Warn for possibly out of range int64 -\> double conversions
  ([\#294](https://github.com/apache/arrow-nanoarrow/issues/294))
- Support map conversion to R vector
  ([\#282](https://github.com/apache/arrow-nanoarrow/issues/282))
- Don’t link to arrow package R6 class pages
  ([\#269](https://github.com/apache/arrow-nanoarrow/issues/269))
- Use [`basic_array_stream()`](../reference/basic_array_stream.md) to
  improve array stream to data.frame conversion
  ([\#279](https://github.com/apache/arrow-nanoarrow/issues/279))

## nanoarrow 0.2.0-1

- Don’t link to arrow package R6 class pages
  ([\#269](https://github.com/apache/arrow-nanoarrow/issues/269))

## nanoarrow 0.2.0

### New features

- Improve printing and conversion of buffers
  ([\#208](https://github.com/apache/arrow-nanoarrow/issues/208))
- Add `enum ArrowType buffer_data_type` member to `struct ArrowLayout`
  ([\#207](https://github.com/apache/arrow-nanoarrow/issues/207))
- Implement ListChildOffset function
  ([\#197](https://github.com/apache/arrow-nanoarrow/issues/197))
- Add ability to deterministically run a finalizer on an array stream
  ([\#196](https://github.com/apache/arrow-nanoarrow/issues/196))
- Union array support
  ([\#195](https://github.com/apache/arrow-nanoarrow/issues/195))
- Add ArrowArrayStream implementation to support keeping a dependent
  object in scope
  ([\#194](https://github.com/apache/arrow-nanoarrow/issues/194))
- Add [`as_nanoarrow_array()`](../reference/as_nanoarrow_array.md)
  implementation that does not fall back on
  [`arrow::as_arrow_array()`](https://arrow.apache.org/docs/r/reference/as_arrow_array.html)
  everywhere
  ([\#108](https://github.com/apache/arrow-nanoarrow/issues/108))
- Create nanoarrow_array objects from buffers
  ([\#105](https://github.com/apache/arrow-nanoarrow/issues/105))
- Implement infer schema methods
  ([\#104](https://github.com/apache/arrow-nanoarrow/issues/104))
- Create and modify nanoarrow_schema objects
  ([\#101](https://github.com/apache/arrow-nanoarrow/issues/101))

### Bugfixes

- Fix [`convert_array_stream()`](../reference/convert_array_stream.md)
  for non-record batch stream with zero batches
  ([\#212](https://github.com/apache/arrow-nanoarrow/issues/212))
- clear `release` in `EmptyArrayStream::release_wrapper`
  ([\#204](https://github.com/apache/arrow-nanoarrow/issues/204))
- Release streams when calling
  [`as.vector()`](https://rdrr.io/r/base/vector.html) or
  [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html)
  ([\#202](https://github.com/apache/arrow-nanoarrow/issues/202))
- Don’t invoke undefined behaviour in conversions to/from Arrow
  ([\#167](https://github.com/apache/arrow-nanoarrow/issues/167))
- Use strict prototypes in all internal C functions
  ([\#151](https://github.com/apache/arrow-nanoarrow/issues/151))
- Don’t memcpy NULL when converting buffer to raw
  ([\#149](https://github.com/apache/arrow-nanoarrow/issues/149))
