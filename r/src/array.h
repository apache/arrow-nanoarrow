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

#ifndef R_NANOARROW_ARRAY_H_INCLUDED
#define R_NANOARROW_ARRAY_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

void finalize_array_xptr(SEXP array_xptr);
void finalize_exported_array(struct ArrowArray* array);

static inline struct ArrowArray* array_from_xptr(SEXP array_xptr) {
  if (!Rf_inherits(array_xptr, "nanoarrow_array")) {
    Rf_error("`array` argument that is not");
  }

  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array == NULL) {
    Rf_error("nanoarrow_array() is an external pointer to NULL");
  }

  if (array->release == NULL) {
    Rf_error("nanoarrow_array() has already been released");
  }

  return array;
}

static inline struct ArrowArray* nullable_array_from_xptr(SEXP array_xptr) {
  if (array_xptr == R_NilValue) {
    return NULL;
  } else {
    return array_from_xptr(array_xptr);
  }
}

static inline SEXP array_owning_xptr() {
  struct ArrowArray* array = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
  array->release = NULL;

  SEXP array_xptr = PROTECT(R_MakeExternalPtr(array, R_NilValue, R_NilValue));
  Rf_setAttrib(array_xptr, R_ClassSymbol, Rf_mkString("nanoarrow_array"));
  R_RegisterCFinalizer(array_xptr, &finalize_array_xptr);
  UNPROTECT(1);
  return array_xptr;
}

static inline void array_xptr_set_schema(SEXP array_xptr, SEXP schema_xptr) {
  R_SetExternalPtrTag(array_xptr, schema_xptr);
}

static inline struct ArrowSchema* schema_from_array_xptr(SEXP array_xptr) {
  SEXP maybe_schema_xptr = R_ExternalPtrTag(array_xptr);
  if (Rf_inherits(maybe_schema_xptr, "nanoarrow_schema")) {
    return (struct ArrowSchema*)R_ExternalPtrAddr(maybe_schema_xptr);
  } else {
    return NULL;
  }
}

static inline void array_export(SEXP array_xptr, struct ArrowArray* array_copy) {
  struct ArrowArray* array = array_from_xptr(array_xptr);

  // keep all the pointers but use the R_PreserveObject mechanism to keep
  // the original data valid (R_ReleaseObject is called from the release callback)
  memcpy(array_copy, array, sizeof(struct ArrowArray));
  array_copy->private_data = array_xptr;
  array_copy->release = &finalize_exported_array;
  R_PreserveObject(array_xptr);
}

// When arrays arrive as a nanoarrow_array, they are responsible for
// releasing their children. This is fine until we need to keep one
// child alive (e.g., a column of a data frame that we attach to an
// ALTREP array) or until we need to export it (i.e., comply with
// https://arrow.apache.org/docs/format/CDataInterface.html#moving-child-arrays
// where child arrays must be movable). To make this work we need to do a shuffle: we
// move the child array to a new owning external pointer and
// give an exported version back to the original object. This only
// applies if the array_xptr has the external pointer 'prot' field
// set (if it doesn't have that set, it is already independent).
static inline SEXP array_ensure_independent(struct ArrowArray* array) {
  SEXP original_array_xptr = PROTECT(array_owning_xptr());

  // Move array to the newly created owner
  struct ArrowArray* original_array = array_from_xptr(original_array_xptr);
  memcpy(original_array, array, sizeof(struct ArrowArray));
  array->release = NULL;

  // Export the independent array (which keeps a reference to original_array_xptr)
  // back to the original home
  array_export(original_array_xptr, array);
  UNPROTECT(1);

  // Return the external pointer of the independent array
  return original_array_xptr;
}

#endif
