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
[![Documentation](https://img.shields.io/badge/Documentation-dev-yellow)](https://apache.github.io/arrow-nanoarrow/dev)
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
CMake or by using the bundled nanoarrow.h/nanorrow.c distribution available in the
dist/ directory in this repository. Examples of both can be found in the examples/
directory in this repository.

A simple producer example:

```c
#include "nanoarrow.h"

int make_simple_array(struct ArrowArray* array_out, struct ArrowSchema* schema_out) {
  struct ArrowError error;
  array_out->release = NULL;
  schema_out->release = NULL;

  NANOARROW_RETURN_NOT_OK(ArrowArrayInit(array_out, NANOARROW_TYPE_INT32));

  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array_out));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 1));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 2));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(array_out, 3));
  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuilding(array_out, &error));
  
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInit(schema_out, NANOARROW_TYPE_INT32));

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
