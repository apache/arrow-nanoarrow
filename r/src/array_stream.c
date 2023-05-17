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
#include "array_stream.h"
#include "nanoarrow.h"
#include "schema.h"
#include "util.h"

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

SEXP nanoarrow_c_basic_array_stream(SEXP batches_sexp, SEXP schema_xptr,
                                    SEXP validate_sexp) {
  int validate = LOGICAL(validate_sexp)[0];

  // Schema needs a copy here because ArrowBasicArrayStreamInit() takes ownership
  SEXP schema_copy_xptr = PROTECT(schema_owning_xptr());
  struct ArrowSchema* schema_copy =
      (struct ArrowSchema*)R_ExternalPtrAddr(schema_copy_xptr);
  schema_export(schema_xptr, schema_copy);

  SEXP array_stream_xptr = PROTECT(array_stream_owning_xptr());
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);

  int64_t n_arrays = Rf_xlength(batches_sexp);
  if (ArrowBasicArrayStreamInit(array_stream, schema_copy, n_arrays) != NANOARROW_OK) {
    Rf_error("Failed to initialize array stream");
  }

  struct ArrowArray array;
  for (int64_t i = 0; i < n_arrays; i++) {
    array_export(VECTOR_ELT(batches_sexp, i), &array);
    ArrowBasicArrayStreamSetArray(array_stream, i, &array);
  }

  if (validate) {
    struct ArrowError error;
    if (ArrowBasicArrayStreamValidate(array_stream, &error) != NANOARROW_OK) {
      Rf_error("ArrowBasicArrayStreamValidate(): %s", ArrowErrorMessage(&error));
    }
  }

  UNPROTECT(2);
  return array_stream_xptr;
}

// Implementation of an ArrowArrayStream that keeps a dependent object valid
struct WrapperArrayStreamData {
  SEXP parent_array_stream_xptr;
  struct ArrowArrayStream* parent_array_stream;
};

static void finalize_wrapper_array_stream(struct ArrowArrayStream* array_stream) {
  if (array_stream->private_data != NULL) {
    struct WrapperArrayStreamData* data =
        (struct WrapperArrayStreamData*)array_stream->private_data;
    nanoarrow_release_sexp(data->parent_array_stream_xptr);
    ArrowFree(array_stream->private_data);
  }

  array_stream->release = NULL;
}

static const char* wrapper_array_stream_get_last_error(
    struct ArrowArrayStream* array_stream) {
  struct WrapperArrayStreamData* data =
      (struct WrapperArrayStreamData*)array_stream->private_data;
  return data->parent_array_stream->get_last_error(data->parent_array_stream);
}

static int wrapper_array_stream_get_schema(struct ArrowArrayStream* array_stream,
                                           struct ArrowSchema* out) {
  struct WrapperArrayStreamData* data =
      (struct WrapperArrayStreamData*)array_stream->private_data;
  return data->parent_array_stream->get_schema(data->parent_array_stream, out);
}

static int wrapper_array_stream_get_next(struct ArrowArrayStream* array_stream,
                                         struct ArrowArray* out) {
  struct WrapperArrayStreamData* data =
      (struct WrapperArrayStreamData*)array_stream->private_data;
  return data->parent_array_stream->get_next(data->parent_array_stream, out);
}

void array_stream_export(SEXP parent_array_stream_xptr,
                         struct ArrowArrayStream* array_stream_copy) {
  struct ArrowArrayStream* parent_array_stream =
      array_stream_from_xptr(parent_array_stream_xptr);

  // If there is no dependent object, don't bother with this wrapper
  SEXP dependent_sexp = R_ExternalPtrProtected(parent_array_stream_xptr);
  if (dependent_sexp == R_NilValue) {
    ArrowArrayStreamMove(parent_array_stream, array_stream_copy);
    return;
  }

  // Allocate a new external pointer for an array stream (for consistency:
  // we always move an array stream when exporting)
  SEXP parent_array_stream_xptr_new = PROTECT(array_stream_owning_xptr());
  struct ArrowArrayStream* parent_array_stream_new =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(parent_array_stream_xptr_new);
  ArrowArrayStreamMove(parent_array_stream, parent_array_stream_new);
  R_SetExternalPtrProtected(parent_array_stream_xptr_new, dependent_sexp);

  array_stream_copy->private_data = NULL;
  array_stream_copy->get_last_error = &wrapper_array_stream_get_last_error;
  array_stream_copy->get_schema = &wrapper_array_stream_get_schema;
  array_stream_copy->get_next = &wrapper_array_stream_get_next;
  array_stream_copy->release = &finalize_wrapper_array_stream;

  struct WrapperArrayStreamData* data =
      (struct WrapperArrayStreamData*)ArrowMalloc(sizeof(struct WrapperArrayStreamData));
  check_trivial_alloc(data, "struct WrapperArrayStreamData");
  data->parent_array_stream_xptr = parent_array_stream_xptr_new;
  data->parent_array_stream = parent_array_stream_new;
  array_stream_copy->private_data = data;

  // Transfer responsibility for the stream_xptr to the C object
  nanoarrow_preserve_sexp(parent_array_stream_xptr_new);
  UNPROTECT(1);
}
