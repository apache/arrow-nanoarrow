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
#include "util.h"
#include "materialize.h"

static void call_as_nanoarrow_array(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr) {
  SEXP fun = PROTECT(Rf_install("as_nanoarrow_array_from_c"));
  SEXP call = PROTECT(Rf_lang3(fun, x_sexp, schema_xptr));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));

  // In many cases we can skip the array_export() step (which adds some complexity
  // and an additional R object to the mix)
  if (Rf_inherits(result, "nanoarrow_array_dont_export")) {
    struct ArrowArray* array_result = array_from_xptr(result);
    ArrowArrayMove(array_result, array);
  } else {
    array_export(result, array);
  }

  UNPROTECT(3);
}

static void as_array_int(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowError* error) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", error->message);
  }

  // Only consider the default create for now
  if (schema_view.type != NANOARROW_TYPE_INT32) {
    call_as_nanoarrow_array(x_sexp, array, schema_xptr);
    return;
  }

  // We don't consider altrep for now: we need an array of int32_t, and while we
  // *could* avoid materializing, there's no point because the source altrep
  // object almost certainly knows how to do this faster than we do.
  int* x_data = INTEGER(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  result = ArrowArrayInitFromType(array, NANOARROW_TYPE_INT32);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  // Borrow the data buffer
  buffer_borrowed(ArrowArrayBuffer(array, 1), x_data, len * sizeof(int32_t), x_sexp);

  // Set the array fields
  array->length = len;
  array->offset = 0;
  int64_t null_count = 0;

  // Look for the first null (will be the last index if there are none)
  int64_t first_null = -1;
  for (int64_t i = 0; i < len; i++) {
    if (x_data[i] == NA_INTEGER) {
      first_null = i;
      break;
    }
  }

  // If there are nulls, pack the validity buffer
  if (first_null != -1) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    ArrowBitmapAppendUnsafe(&bitmap, 1, first_null);
    for (int64_t i = first_null; i < len; i++) {
      uint8_t is_valid = x_data[i] != NA_INTEGER;
      null_count+= !is_valid;
      ArrowBitmapAppend(&bitmap, is_valid, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }

  array->null_count = null_count;
  result = ArrowArrayFinishBuilding(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuilding(): %s", error->message);
  }
}

static void as_array_dbl(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowError* error) {
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", error->message);
  }

  // Only consider the default create for now
  if (schema_view.type != NANOARROW_TYPE_DOUBLE) {
    call_as_nanoarrow_array(x_sexp, array, schema_xptr);
    return;
  }

  double* x_data = REAL(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  result = ArrowArrayInitFromType(array, NANOARROW_TYPE_DOUBLE);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  // Borrow the data buffer
  buffer_borrowed(ArrowArrayBuffer(array, 1), x_data, len * sizeof(double), x_sexp);

  // Set the array fields
  array->length = len;
  array->offset = 0;
  int64_t null_count = 0;

  // Look for the first null (will be the last index if there are none)
  int64_t first_null = -1;
  for (int64_t i = 0; i < len; i++) {
    if (R_IsNA(x_data[i]) || R_IsNaN(x_data[i])) {
      first_null = i;
      break;
    }
  }

  // If there are nulls, pack the validity buffer
  if (first_null != -1) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    ArrowBitmapAppendUnsafe(&bitmap, 1, first_null);
    for (int64_t i = first_null; i < len; i++) {
      uint8_t is_valid = !R_IsNA(x_data[i]) && !R_IsNaN(x_data[i]);
      null_count+= !is_valid;
      ArrowBitmapAppend(&bitmap, is_valid, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }

  array->null_count = null_count;
  result = ArrowArrayFinishBuilding(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuilding(): %s", error->message);
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
    call_as_nanoarrow_array(x_sexp, array, schema_xptr);
    return;
  }

  if (Rf_xlength(x_sexp) != schema->n_children) {
    Rf_error("Expected %ld schema children for data.frame with %ld columns",
             (long)schema->n_children, (long)Rf_xlength(x_sexp));
  }

  result = ArrowArrayInitFromType(array, NANOARROW_TYPE_STRUCT);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  result = ArrowArrayAllocateChildren(array, schema->n_children);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayAllocateChildren() failed");
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    as_array_default(VECTOR_ELT(x_sexp, i), array->children[i],
                     borrow_schema_child_xptr(schema_xptr, i), error);
  }

  array->length = nanoarrow_data_frame_size(x_sexp);
  array->null_count = 0;
  array->offset = 0;
}

static void as_array_default(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowError* error) {
  if (Rf_isObject(x_sexp)) {
    if (Rf_inherits(x_sexp, "data.frame")) {
      as_array_data_frame(x_sexp, array, schema_xptr, error);
      return;
    } else {
      call_as_nanoarrow_array(x_sexp, array, schema_xptr);
      return;
    }
  }

  switch (TYPEOF(x_sexp)) {
    case INTSXP:
      as_array_int(x_sexp, array, schema_xptr, error);
      return;
    case REALSXP:
      as_array_dbl(x_sexp, array, schema_xptr, error);
      return;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr);
      return;
  }
}

SEXP nanoarrow_c_as_array_default(SEXP x_sexp, SEXP schema_sexp) {
  SEXP array_xptr = PROTECT(array_owning_xptr());
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  struct ArrowError error;
  as_array_default(x_sexp, array, schema_sexp, &error);
  array_xptr_set_schema(array_xptr, schema_sexp);
  UNPROTECT(1);
  return array_xptr;
}
