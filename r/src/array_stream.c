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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "array_stream.h"
#include "schema.h"
#include "array.h"
#include "nanoarrow.h"

void finalize_array_stream_xptr(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream != NULL && array_stream->release != NULL) {
    array_stream->release(array_stream);
  }

  if (array_stream != NULL) {
    ArrowFree(array_stream);
  }
}

SEXP nanoarrow_c_array_stream_get_schema(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream = array_stream_from_xptr(array_stream_xptr);

  SEXP schema_xptr = PROTECT(schema_owning_xptr());
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  int result = array_stream->get_schema(array_stream, schema);

  if (result != 0) {
    const char* last_error = array_stream->get_last_error(array_stream);
    if (last_error == NULL) {
      last_error = "";
    }
    Rf_error("array_stream->get_schema(): [%d] %s", result, last_error);
  }

  UNPROTECT(1);
  return schema_xptr;
}

SEXP nanoarrow_c_array_stream_get_next(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream = array_stream_from_xptr(array_stream_xptr);

  SEXP array_xptr = PROTECT(array_owning_xptr());
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  int result = array_stream->get_next(array_stream, array);
  
  if (result != 0) {
    const char* last_error = array_stream->get_last_error(array_stream);
    if (last_error == NULL) {
      last_error = "";
    }
    Rf_error("array_stream->get_next(): [%d] %s", result, last_error);
  }

  UNPROTECT(1);
  return array_xptr;
}
