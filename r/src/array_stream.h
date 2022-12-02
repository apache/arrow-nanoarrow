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

#ifndef R_NANOARROW_ARRAY_STREAM_H_INCLUDED
#define R_NANOARROW_ARRAY_STREAM_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"
#include "util.h"

void finalize_array_stream_xptr(SEXP array_stream_xptr);

// Returns the underlying struct ArrowSchema* from an external pointer,
// checking and erroring for invalid objects, pointers, and arrays.
static inline struct ArrowArrayStream* array_stream_from_xptr(SEXP array_stream_xptr) {
  if (!Rf_inherits(array_stream_xptr, "nanoarrow_array_stream")) {
    Rf_error("`array_stream` argument that is not a nanoarrow_array_stream()");
  }

  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream == NULL) {
    Rf_error("nanoarrow_array_stream() is an external pointer to NULL");
  }

  if (array_stream->release == NULL) {
    Rf_error("nanoarrow_array_stream() has already been released");
  }

  return array_stream;
}

// Create an external pointer with the proper class and that will release any
// non-null, non-released pointer when garbage collected.
static inline SEXP array_stream_owning_xptr(void) {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)ArrowMalloc(sizeof(struct ArrowArrayStream));
  array_stream->release = NULL;

  SEXP array_stream_xptr =
      PROTECT(R_MakeExternalPtr(array_stream, R_NilValue, R_NilValue));
  Rf_setAttrib(array_stream_xptr, R_ClassSymbol, nanoarrow_cls_array_stream);
  R_RegisterCFinalizer(array_stream_xptr, &finalize_array_stream_xptr);
  UNPROTECT(1);
  return array_stream_xptr;
}

#endif
