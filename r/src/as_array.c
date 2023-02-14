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
#include "buffer.h"
#include "nanoarrow.h"
#include "schema.h"

static void as_array_int(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowError* error) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", error->message);
  }

  // We don't consider altrep for now: we need an array of int32_t, and while we
  // *could* avoid materializing, there's no point because the source altrep
  // object almost certainly knows how to do this faster than we do.
  // Doing this first because it may jump.
  int* x_data = INTEGER(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  result = ArrowArrayInitFromType(array, NANOARROW_TYPE_INT32);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  // Borrow the data buffer
  buffer_borrowed(ArrowArrayBuffer(array, 1), x_data, len * sizeof(int32_t), x_sexp);

  // Look for the first null (will be the last index if there are none)
  int64_t first_null = 0;
  for (int64_t i = 0; i < len; i++) {
    if (x_data[i] == NA_INTEGER) {
      first_null = i;
      break;
    }
  }

  // If there are nulls, pack the validity buffer
  if (first_null < (len - 1)) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    ArrowBitmapAppendUnsafe(&bitmap, 0, first_null);
    for (int64_t i = first_null; i < len; i++) {
      ArrowBitmapAppendUnsafe(&bitmap, x_data[i] != NA_INTEGER, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }
}

static void as_array_default(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowError* error);

static void as_array_data_frame(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                                struct ArrowError* error) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", error->message);
  }

  if (schema_view.type != NANOARROW_TYPE_STRUCT) {
    Rf_error("Arrow type not supported");
  }

  if (Rf_xlength(x_sexp) != schema->n_children) {
    Rf_error("Expected %ld schema children for data.frame with %ld columns",
             (long)schema->n_children, (long)Rf_xlength(x_sexp));
  }

  result = ArrowArrayAllocateChildren(array, schema->n_children);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayAllocateChildren() failed");
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    as_array_default(VECTOR_ELT(x_sexp, i), array->children[i],
                     borrow_schema_child_xptr(schema_xptr, i), error);
  }
}

static void as_array_default(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowError* error) {
  if (Rf_isObject(x_sexp)) {
    if (Rf_inherits(x_sexp, "data.frame")) {
      as_array_data_frame(x_sexp, array, schema_xptr, error);
      return;
    } else {
      Rf_error("Can't convert: S3 type not supported");
    }
  }

  switch (TYPEOF(x_sexp)) {
    case INTSXP:
      as_array_int(x_sexp, array, schema_xptr, error);
      return;
    default:
      Rf_error("Can't convert: type not supported");
  }
}

SEXP nanoarrow_c_as_array_default(SEXP x_sexp, SEXP schema_sexp) {
  SEXP array_xptr = PROTECT(array_owning_xptr());
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  struct ArrowError error;
  as_array_default(x_sexp, array, schema_sexp, &error);
  UNPROTECT(1);
  return array_xptr;
}
