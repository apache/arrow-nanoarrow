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

#include "array.h"
#include "nanoarrow.h"
#include "schema.h"

void finalize_array_xptr(SEXP array_xptr) {
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array != NULL && array->release != NULL) {
    array->release(array);
  }
}

SEXP nanoarrow_c_array_set_schema(SEXP array_xptr, SEXP schema_xptr) {
  // Fair game to remove a schema from a pointer
  if (schema_xptr == R_NilValue) {
    array_xptr_set_schema(array_xptr, R_NilValue);
    return R_NilValue;
  }

  // If adding a schema, validate the pair
  struct ArrowArray* array = array_from_xptr(array_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  struct ArrowArrayView array_view;
  struct ArrowError error;
  int result = ArrowArrayViewInitFromSchema(&array_view, schema, &error);
  if (result != NANOARROW_OK) {
    Rf_error("%s", ArrowErrorMessage(&error));
  }

  result = ArrowArrayViewSetArray(&array_view, array, &error);
  if (result != NANOARROW_OK) {
    Rf_error("%s", ArrowErrorMessage(&error));
  }

  array_xptr_set_schema(array_xptr, schema_xptr);
  return R_NilValue;
}

SEXP nanoarrow_c_infer_schema_array(SEXP array_xptr) {
  SEXP maybe_schema_xptr = R_ExternalPtrTag(array_xptr);
  if (Rf_inherits(maybe_schema_xptr, "nanoarrow_schema")) {
    return maybe_schema_xptr;
  } else {
    return R_NilValue;
  }
}

// for ArrowArray* that are exported references to an R schema_xptr
void finalize_exported_array(struct ArrowArray* array) {
  SEXP array_xptr = (SEXP) array->private_data;
  R_ReleaseObject(array_xptr);

  // TODO: properly relocate child arrays
  // https://arrow.apache.org/docs/format/CDataInterface.html#moving-child-arrays

  array->release = NULL;
}
