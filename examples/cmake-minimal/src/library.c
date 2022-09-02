// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <errno.h>
#include <stdlib.h>

#include "nanoarrow/nanoarrow.h"

#include "library.h"

static struct ArrowError my_library_last_error_;

const char* my_library_last_error() { return ArrowErrorMessage(&my_library_last_error_); }

int my_library_int32_array_from_args(int n_args, char* argv[],
                                     struct ArrowArray* array_out,
                                     struct ArrowSchema* schema_out) {
  ArrowErrorSet(&my_library_last_error_, "");

  int result = ArrowArrayInit(array_out, NANOARROW_TYPE_INT32);
  if (result != NANOARROW_OK) {
    return result;
  }

  result = ArrowArrayStartAppending(array_out);
  if (result != NANOARROW_OK) {
    array_out->release(array_out);
    return result;
  }

  char* end_char;
  for (int i = 0; i < n_args; i++) {
    int64_t value = strtol(argv[i], &end_char, 10);
    if (end_char != (argv[i] + strlen(argv[i]))) {
      ArrowErrorSet(&my_library_last_error_, "Can't parse argument %d ('%s') to long int",
                    i + 1, argv[i]);
      array_out->release(array_out);
      return EINVAL;
    }

    result = ArrowArrayAppendInt(array_out, value);
    if (result != NANOARROW_OK) {
      ArrowErrorSet(&my_library_last_error_,
                    "Error appending argument %d ('%s') to array", i + 1, argv[i]);
      array_out->release(array_out);
      return result;
    }
  }

  result = ArrowArrayFinishBuilding(array_out, &my_library_last_error_);
  if (result != NANOARROW_OK) {
    return result;
  }

  result = ArrowSchemaInit(schema_out, NANOARROW_TYPE_INT32);
  if (result != NANOARROW_OK) {
    array_out->release(array_out);
    return result;
  }

  return NANOARROW_OK;
}
