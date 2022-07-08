
# nanoarrow

[![Codecov test coverage](https://codecov.io/gh/paleolimbot/nanoarrow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/paleolimbot/nanoarrow?branch=main)

The nanoarrow library is a set of helper functions to interpret and generate
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
and [Arrow C Stream Interface](https://arrow.apache.org/docs/format/CStreamInterface.html)
structures. The library is in active development and should currently be used only
for entertainment purposes. Everything from the name of the project to the variable
names are up for grabs (i.e., suggest/pull request literally any ideas you may
have!).

Whereas the current suite of Arrow implementations provide the basis for a
comprehensive data analysis toolkit, this library is intended to support clients
that wish to produce or interpret Arrow C Data and/or Arrow C Stream structures.
The library will:

- Create, copy, parse, and validate struct ArrowSchema objects (all types mentioned
  in the C Data interface specification)
- Create and validate struct ArrowArray/struct ArrowSchema pairs for (all types
  mentioned in the C Data interface specification)
- Iterate over struct ArrowArrays element-wise (non-nested types) (i.e., is the
  ith element null; get the ith element).
- Build Arrays element-wise (non-nested types) (i.e., basic Array Builder logic).

While it will not provide full support for nested types, it should provide enough
infrastructure that an extension library with a similar format could implement such
support.

## Usage

You can use nanoarrow in your project in two ways:

1. Copy contents of the `src/nanoarrow/` into your favourite include directory and
   `#include <nanoarrow/nanoarrow.c>` somewhere in your project exactly once.
2. Clone and use `cmake`, `cmake --build`, and `cmake --install` to build/install
   the static library and add `-L/path/to/nanoarrow/lib -lnanoarrow` to your favourite
   linker flag configuration.

All public functions and types are declared in `nanoarrow/nanoarrow.h`.

In all cases you will want to copy this project or pin your build to a specific commit
since it will change rapidly and regularly. The nanoarrow library does not and will
not provide ABI stability (i.e., you must vendor or link to a private version of
the static library).

## Background

The design of nanoarrow reflects the needs of a few previous libraries/prototypes
requiring a library with a similar scope:

- DuckDB’s Arrow wrappers, the details of which are in a few places
  (e.g., [here](https://github.com/duckdb/duckdb/blob/master/src/common/arrow_wrapper.cpp),
  [here](https://github.com/duckdb/duckdb/blob/master/src/main/query_result.cpp),
  and a few other places)
- An [R wrapper around the C Data interface](https://github.com/paleolimbot/narrow),
  along which a [C-only library](https://github.com/paleolimbot/narrow/tree/master/src/narrow)
  was prototyped.
- An [R implementation of the draft GeoArrow specification](https://github.com/paleolimbot/geoarrow),
  along which a [mostly header-only C++ library](https://github.com/paleolimbot/geonanoarrowpp/tree/main/src/geoarrow/internal/arrow-hpp)
  was prototyped.
- The [Arrow Database Connector](https://github.com/apache/arrow-adbc) C API, for which drivers
  in theory can be written in C (which is currently difficult in practice because of there
  are few if any tools to help do this properly).
