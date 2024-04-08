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
[![Documentation](https://img.shields.io/badge/Documentation-main-yellow)](https://arrow.apache.org/nanoarrow/main)
[![nanoarrow on GitHub](https://img.shields.io/badge/GitHub-apache%2Farrow--nanoarrow-blue)](https://github.com/apache/arrow-nanoarrow)

The nanoarrow library is a set of helper functions to interpret and generate
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)
and [Arrow C Stream Interface](https://arrow.apache.org/docs/format/CStreamInterface.html)
structures. The library is in active early development and users should update regularly
from the main branch of this repository.

Whereas the current suite of Arrow implementations provide the basis for a
comprehensive data analysis toolkit, this library is intended to support clients
that wish to produce or interpret Arrow C Data and/or Arrow C Stream structures
where linking to a higher level Arrow binding is difficult or impossible.

## Using the C library

The nanoarrow C library is intended to be copied and vendored. This can be done using
CMake or by using the bundled nanoarrow.h/nanoarrow.c distribution available in the
dist/ directory in this repository. Examples of both can be found in the examples/
directory in this repository.

A simple producer example:

```c
#include "nanoarrow.h"

int make_simple_array(struct ArrowArray* array_out, struct ArrowSchema* schema_out) {
  struct ArrowError error;
  array_out->release = NULL;
  schema_out->release = NULL;

  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array_out, NANOARROW_TYPE_INT32));

  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array_out));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 1));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 2));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 3));
  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(array_out, &error));

  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema_out, NANOARROW_TYPE_INT32));

  return NANOARROW_OK;
}
```

A simple consumer example:

```c
#include <stdio.h>

#include "nanoarrow.h"

int print_simple_array(struct ArrowArray* array, struct ArrowSchema* schema) {
  struct ArrowError error;
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(&array_view, schema, &error));

  if (array_view.storage_type != NANOARROW_TYPE_INT32) {
    printf("Array has storage that is not int32\n");
  }

  int result = ArrowArrayViewSetArray(&array_view, array, &error);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(&array_view);
    return result;
  }

  for (int64_t i = 0; i < array->length; i++) {
    printf("%d\n", (int)ArrowArrayViewGetIntUnsafe(&array_view, i));
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}
```

## Building with Meson

CMake is the officially supported build system for nanoarrow. However, the Meson backend is an experimental feature you may also wish to try.

To run the test suite with Meson, you will want to first install the testing dependencies via the wrap database (n.b. no wrap database entry exists for Arrow - that must be installed separately).

```sh
mkdir subprojects
meson wrap install gtest
meson wrap install google-benchmark
meson wrap install nlohmann_json
```

The Arrow C++ library must also be discoverable via pkg-config build tests.

You can then set up your build directory:

```sh
meson setup builddir
cd builddir
```

And configure your project (this could have also been done inline with ``setup``)

```sh
meson configure -DNANOARROW_BUILD_TESTS=true -DNANOARROW_BUILD_BENCHMARKS=true
```

Note that if your Arrow pkg-config profile is installed in a non-standard location on your system, you may pass the ``--pkg-config-path <path to directory with arrow.pc>`` to either the setup or configure steps above.

With the above out of the way, the ``compile`` command should take care of the rest:

```sh
meson compile
```

Upon a successful build you can execute the test suite and benchmarks with the following commands:

```sh
meson test nanoarrow:  # default test run
meson test nanoarrow: --wrap valgrind  # run tests under valgrind
meson test nanoarrow: --benchmark --verbose # run benchmarks
```
