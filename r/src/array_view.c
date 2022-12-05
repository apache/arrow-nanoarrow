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

#include "nanoarrow.h"

#include "array.h"
#include "schema.h"
#include "util.h"

static void finalize_array_view_xptr(SEXP array_view_xptr) {
  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  if (array_view != NULL) {
    ArrowArrayViewReset(array_view);
    ArrowFree(array_view);
  }
}

SEXP nanoarrow_c_array_view(SEXP array_xptr, SEXP schema_xptr) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  struct ArrowError error;
  ArrowErrorSet(&error, "");

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
  ArrowArrayViewInitFromType(array_view, NANOARROW_TYPE_UNINITIALIZED);
  SEXP xptr = PROTECT(R_MakeExternalPtr(array_view, R_NilValue, array_xptr));
  R_RegisterCFinalizer(xptr, &finalize_array_view_xptr);

  int result = ArrowArrayViewInitFromSchema(array_view, schema, &error);
  if (result != NANOARROW_OK) {
    Rf_error("<ArrowArrayViewInitFromSchema> %s", error.message);
  }

  result = ArrowArrayViewSetArray(array_view, array, &error);
  if (result != NANOARROW_OK) {
    Rf_error("<ArrowArrayViewSetArray> %s", error.message);
  }

  Rf_setAttrib(xptr, R_ClassSymbol, nanoarrow_cls_array_view);
  UNPROTECT(1);
  return xptr;
}

SEXP array_view_xptr_from_array_xptr(SEXP array_xptr) {
  return nanoarrow_c_array_view(array_xptr, R_ExternalPtrTag(array_xptr));
}
