<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# nanoarrow

[![Codecov test coverage](https://codecov.io/gh/apache/arrow-nanoarrow/branch/main/graph/badge.svg)](https://app.codecov.io/gh/apache/arrow-nanoarrow?branch=main)
[![Documentation](https://img.shields.io/badge/Documentation-main-yellow)](https://arrow.apache.org/nanoarrow/)
[![nanoarrow on GitHub](https://img.shields.io/badge/GitHub-apache%2Farrow--nanoarrow-blue)](https://github.com/apache/arrow-nanoarrow)

The nanoarrow libraries are a set of helpers to produce and consume Arrow data,
including the
[Arrow C Data](https://arrow.apache.org/docs/format/CDataInterface.html),
[Arrow C Stream](https://arrow.apache.org/docs/format/CStreamInterface.html),
and [Arrow C Device](https://arrow.apache.org/docs/format/CDeviceDataInterface.html),
structures and the
[serialized Arrow IPC format](https://arrow.apache.org/docs/format/Columnar.html#serialization-and-interprocess-communication-ipc).
The vision of nanoarrow is that it should be trivial for libraries to produce and consume
Arrow data: it helps fulfill this vision by providing high-quality, easy-to-adopt
helpers to produce, consume, and test Arrow data types and arrays.

The nanoarrow libraries were built to be:

- Small: nanoarrow’s C runtime compiles into a few hundred kilobytes and its R and Python
  bindings both have an installed size of ~1 MB.
- Easy to depend on: nanoarrow's C library is distributed as two files (nanoarrow.c and
  nanoarrow.h) and its R and Python bindings have zero dependencies.
- Useful: The Arrow Columnar Format includes a wide range of data type and data encoding
  options. To the greatest extent practicable, nanoarrow strives to support the entire
  Arrow columnar specification (see the
  [Arrow implementation status](https://arrow.apache.org/docs/status.html) page for
  implementation status).

## Getting started

The nanoarrow Python bindings are available from [PyPI](https://pypi.org/) and
[conda-forge](https://conda-forge.org/):

```sh
pip install nanoarrow
conda install nanoarrow -c conda-forge
```

The nanoarrow R package is available from [CRAN](https://cran.r-project.org):

```r
install.packages("nanoarrow")
```

The C library can be used by generating bundled versions of the core library
and its components. This is the version used internally by the R and Python
bindings.

```shell
python ci/scripts/bundle.py \
  --source-output-dir=dist \
  --include-output-dir=dist \
  --header-namespace= \
  --with-device \
  --with-ipc \
  --with-testing \
  --with-flatcc
```

CMake is also supported via a build/install with `find_package()` or using
`FetchContent`:

```
fetchcontent_declare(nanoarrow
                     URL "https://www.apache.org/dyn/closer.lua?action=download&filename=arrow/apache-arrow-nanoarrow-0.6.0/apache-arrow-nanoarrow-0.6.0.tar.gz")

fetchcontent_makeavailable(nanoarrow)
```

The C library can also be used as a Meson subproject installed with:

```shell
mkdir subprojects
meson wrap install nanoarrow
```

...and declared as a dependency with:

```
nanoarrow_dep = dependency('nanoarrow')
example_exec = executable('example_meson_minimal_app',
                          'src/app.cc',
                          dependencies: [nanoarrow_dep])
```

See the [nanoarrow Documentation](https://arrow.apache.org/nanoarrow/latest/) for
extended tutorials and API reference for the C, C++, Python, and R libraries.

- [Getting started in C/C++](https://arrow.apache.org/nanoarrow/latest/getting-started/cpp.html)
- [Getting started in Python](https://arrow.apache.org/nanoarrow/latest/getting-started/python.html)
- [Getting started in R](https://arrow.apache.org/nanoarrow/latest/getting-started/r.html)

The [nanoarrow GitHub repository](https://github.com/apache/arrow-nanoarrow) additionally
provides a number of [examples](https://github.com/apache/arrow-nanoarrow/tree/main/examples)
covering how to use nanoarrow in a variety of build configurations.

## Development

### Building with CMake

CMake is the primary build system used to develop and test the nanoarrow C library. You can build
nanoarrow with:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

To build nanoarrow along with tests run:

```sh
mkdir build && cd build
cmake .. -DNANOARROW_BUILD_TESTS=ON
cmake --build .
```

If you are able to install [Arrow C++](https://arrow.apache.org/install/) you can enable
more testing:

```sh
mkdir build && cd build
cmake .. -DNANOARROW_BUILD_TESTS=ON -DNANOARROW_BUILD_TESTS_WITH_ARROW=ON
cmake --build .
```

Tests can be run with `ctest`.

### Building with Meson

CMake is the officially supported build system for nanoarrow. However, the Meson
backend is an experimental feature you may also wish to try.

```sh
meson setup builddir
cd builddir
```

After setting up your project, be sure to enable the options you want:

```sh
meson configure -Dtests=enabled -Dbenchmarks=enabled
```

You can enable better test coverage if Apache Arrow is installed on your system
with `-Dtest_with_arrow=enabled`. Depending on how you have installed Apache Arrow,
you may also need to pass `--pkg-config-path <path to directory with arrow.pc>`.

With the above out of the way, the `compile` command should take care of the rest:

```sh
meson compile
```

Upon a successful build you can execute the test suite and benchmarks with the
following commands:

```sh
meson test nanoarrow:  # default test run
meson test nanoarrow: --wrap valgrind  # run tests under valgrind
meson test nanoarrow: --benchmark --verbose # run benchmarks
```
