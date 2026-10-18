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
#include <errno.h>
#include <string.h>

#include "array.h"
#include "array_stream.h"
#include "nanoarrow.h"
#include "schema.h"
#include "util.h"

struct FunctionArrayStreamData {
  struct ArrowSchema schema;
  SEXP fun;
  SEXP schema_xptr;
  int validate;
  struct ArrowError error;
  int error_code;
  int finished;
  int errored;
  int evaluating;
  int release_pending;
};

struct FunctionArrayStreamCall {
  struct FunctionArrayStreamData* data;
  struct ArrowArray* out;
  int code;
  int caught;
};

static void release_function_array_stream_data(struct FunctionArrayStreamData* data) {
  if (data->schema.release != NULL) {
    data->schema.release(&data->schema);
  }
  nanoarrow_release_sexp(data->fun);
  nanoarrow_release_sexp(data->schema_xptr);
  ArrowFree(data);
}

static void function_array_stream_set_error(struct FunctionArrayStreamData* data,
                                            int code, const char* message) {
  if (!data->errored) {
    ArrowErrorSet(&data->error, "%s", message);
    data->error_code = code;
    data->errored = 1;
  }
}

static void release_function_array_stream(struct ArrowArrayStream* stream) {
  struct FunctionArrayStreamData* data =
      (struct FunctionArrayStreamData*)stream->private_data;
  stream->release = NULL;
  stream->private_data = NULL;
  if (data->evaluating) {
    // The callback may release its own stream. Keep its state alive until it returns.
    data->release_pending = 1;
    return;
  }
  release_function_array_stream_data(data);
}

static const char* function_array_stream_get_last_error(struct ArrowArrayStream* stream) {
  struct FunctionArrayStreamData* data =
      (struct FunctionArrayStreamData*)stream->private_data;
  return data == NULL ? "Array stream has been released"
                      : ArrowErrorMessage(&data->error);
}

static int function_array_stream_get_schema(struct ArrowArrayStream* stream,
                                            struct ArrowSchema* out) {
  struct FunctionArrayStreamData* data =
      (struct FunctionArrayStreamData*)stream->private_data;
  return ArrowSchemaDeepCopy(&data->schema, out);
}

static SEXP function_array_stream_condition_handler(SEXP condition, void* opaque) {
  struct FunctionArrayStreamCall* call = (struct FunctionArrayStreamCall*)opaque;
  call->caught = 1;
  if (Rf_inherits(condition, "interrupt")) {
    call->caught = 2;
  }
  SEXP message_call = PROTECT(Rf_lang2(Rf_install("conditionMessage"), condition));
  SEXP message = Rf_eval(message_call, R_BaseEnv);
  UNPROTECT(1);
  return message;
}

static SEXP function_array_stream_eval_call(void* opaque) {
  return Rf_eval((SEXP)opaque, nanoarrow_ns_pkg);
}

static void function_array_stream_eval(void* opaque) {
  struct FunctionArrayStreamCall* call = (struct FunctionArrayStreamCall*)opaque;
  struct FunctionArrayStreamData* data = call->data;
  SEXP classes = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(classes, 0, Rf_mkChar("error"));
  SET_STRING_ELT(classes, 1, Rf_mkChar("interrupt"));
  SEXP validate = PROTECT(Rf_ScalarLogical(data->validate));
  SEXP callback = PROTECT(Rf_lang4(Rf_install("function_array_stream_get_next_wrapper"),
                                   data->fun, data->schema_xptr, validate));
  SEXP result =
      PROTECT(R_tryCatch(&function_array_stream_eval_call, callback, classes,
                         &function_array_stream_condition_handler, call, NULL, NULL));
  if (data->errored) {
    call->code = data->error_code;
  } else if (call->caught) {
    const char* message = Rf_translateCharUTF8(STRING_ELT(result, 0));
    if (call->caught == 2) {
      struct ArrowError error;
      ArrowErrorSet(&error, "R callback interrupted: %s", message);
      function_array_stream_set_error(data, EIO, ArrowErrorMessage(&error));
    } else {
      function_array_stream_set_error(data, EIO, message);
    }
    call->code = data->error_code;
  } else if (result == R_NilValue) {
    data->finished = 1;
    call->code = NANOARROW_OK;
  } else {
    // Let the external pointer clean up a partial export if R jumps during allocation.
    SEXP array_xptr = PROTECT(nanoarrow_array_owning_xptr());
    struct ArrowArray* array = nanoarrow_output_array_from_xptr(array_xptr);
    array_export(result, array);
    ArrowArrayMove(array, call->out);
    UNPROTECT(1);
    call->code = NANOARROW_OK;
  }
  UNPROTECT(4);
}

static int function_array_stream_get_next(struct ArrowArrayStream* stream,
                                          struct ArrowArray* out) {
  struct FunctionArrayStreamData* data =
      (struct FunctionArrayStreamData*)stream->private_data;
  out->release = NULL;
  if (data == NULL || data->finished) {
    return NANOARROW_OK;
  }
  if (data->errored) {
    return data->error_code;
  }
  if (!nanoarrow_is_main_thread()) {
    function_array_stream_set_error(data, EIO,
                                    "Can't evaluate R callback on a non-R thread");
    return data->error_code;
  }
  if (data->evaluating) {
    function_array_stream_set_error(data, EINVAL,
                                    "Re-entrant call to function array stream");
    return data->error_code;
  }

  data->evaluating = 1;
  struct FunctionArrayStreamCall call = {data, out, NANOARROW_OK, 0};
  // R_tryCatch() does not catch restart jumps or failures in conditionMessage().
  // All R API calls must stay inside this outer boundary.
  if (!R_ToplevelExec(&function_array_stream_eval, &call)) {
    function_array_stream_set_error(
        data, EIO, "R callback exited non-locally (e.g., via an abort restart)");
    call.code = data->error_code;
  }
  data->evaluating = 0;

  if (data->release_pending) {
    if (out->release != NULL) {
      out->release(out);
    }
    out->release = NULL;
    // The stream itself may already have been freed by a finalizer during evaluation.
    release_function_array_stream_data(data);
    return EINVAL;
  }
  return call.code;
}

SEXP nanoarrow_c_function_array_stream(SEXP fun, SEXP schema_xptr, SEXP validate) {
  SEXP stream_xptr = PROTECT(nanoarrow_array_stream_owning_xptr());
  struct ArrowArrayStream* stream = nanoarrow_output_array_stream_from_xptr(stream_xptr);
  SEXP schema_copy_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema_copy = nanoarrow_output_schema_from_xptr(schema_copy_xptr);
  schema_export(schema_xptr, schema_copy);
  struct FunctionArrayStreamData* data =
      (struct FunctionArrayStreamData*)ArrowMalloc(sizeof(*data));
  check_trivial_alloc(data, "struct FunctionArrayStreamData");
  memset(data, 0, sizeof(*data));
  ArrowErrorInit(&data->error);
  data->fun = R_NilValue;
  data->schema_xptr = R_NilValue;
  data->validate = LOGICAL(validate)[0];
  stream->private_data = data;
  stream->get_last_error = &function_array_stream_get_last_error;
  stream->get_schema = &function_array_stream_get_schema;
  stream->get_next = &function_array_stream_get_next;
  stream->release = &release_function_array_stream;
  // Attach the finalizer before any operation that can allocate or fail.
  nanoarrow_preserve_sexp(fun);
  data->fun = fun;
  nanoarrow_preserve_sexp(schema_copy_xptr);
  data->schema_xptr = schema_copy_xptr;
  schema_export(schema_copy_xptr, &data->schema);
  UNPROTECT(2);
  return stream_xptr;
}

// Ideally user-supplied finalizers are written in such a way that they don't jump;
// however if they do it is likely that memory will leak. Here, we use
// R_tryCatchError to minimize the chances of that happening.
static SEXP run_finalizer_wrapper(void* data) {
  SEXP finalizer_sym = PROTECT(Rf_install("array_stream_finalizer"));
  SEXP finalizer_call = PROTECT(Rf_lang1(finalizer_sym));
  Rf_eval(finalizer_call, (SEXP)data);
  UNPROTECT(2);
  return R_NilValue;
}

static SEXP run_finalizer_error_handler(SEXP cond, void* hdata) {
  REprintf("Error evaluating user-supplied array stream finalizer");
  return R_NilValue;
}

static void run_user_array_stream_finalizer(SEXP array_stream_xptr) {
  SEXP protected = PROTECT(R_ExternalPtrProtected(array_stream_xptr));
  R_SetExternalPtrProtected(array_stream_xptr, R_NilValue);

  if (Rf_inherits(protected, "nanoarrow_array_stream_finalizer")) {
    R_tryCatchError(&run_finalizer_wrapper, protected, &run_finalizer_error_handler,
                    NULL);
  }

  UNPROTECT(1);
}

SEXP nanoarrow_c_array_stream_get_schema(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream =
      nanoarrow_array_stream_from_xptr(array_stream_xptr);

  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  int result = ArrowArrayStreamGetSchema(array_stream, schema, NULL);
  if (result != 0) {
    Rf_error("array_stream->get_schema(): [%d] %s", result,
             ArrowArrayStreamGetLastError(array_stream));
  }

  UNPROTECT(1);
  return schema_xptr;
}

SEXP nanoarrow_c_array_stream_get_next(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream =
      nanoarrow_array_stream_from_xptr(array_stream_xptr);

  SEXP array_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* array = nanoarrow_output_array_from_xptr(array_xptr);

  int result = ArrowArrayStreamGetNext(array_stream, array, NULL);
  if (result != NANOARROW_OK) {
    Rf_error("array_stream->get_next(): [%d] %s", result,
             ArrowArrayStreamGetLastError(array_stream));
  }

  UNPROTECT(1);
  return array_xptr;
}

SEXP nanoarrow_c_basic_array_stream(SEXP batches_sexp, SEXP schema_xptr,
                                    SEXP validate_sexp) {
  int validate = LOGICAL(validate_sexp)[0];

  // Schema needs a copy here because ArrowBasicArrayStreamInit() takes ownership
  SEXP schema_copy_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema_copy = nanoarrow_output_schema_from_xptr(schema_copy_xptr);
  schema_export(schema_xptr, schema_copy);

  SEXP array_stream_xptr = PROTECT(nanoarrow_array_stream_owning_xptr());
  struct ArrowArrayStream* array_stream =
      nanoarrow_output_array_stream_from_xptr(array_stream_xptr);

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

SEXP nanoarrow_c_array_list_total_length(SEXP list_of_array_xptr) {
  int64_t total_length = 0;
  R_xlen_t num_chunks = Rf_xlength(list_of_array_xptr);
  for (R_xlen_t i = 0; i < num_chunks; i++) {
    struct ArrowArray* chunk =
        (struct ArrowArray*)R_ExternalPtrAddr(VECTOR_ELT(list_of_array_xptr, i));
    total_length += chunk->length;
  }

  return length_sexp_from_int64(total_length);
}

// Implementation of an ArrowArrayStream that keeps a dependent object valid
struct WrapperArrayStreamData {
  SEXP parent_array_stream_xptr;
  struct ArrowArrayStream* parent_array_stream;
};

static void finalize_wrapper_array_stream(struct ArrowArrayStream* array_stream) {
  struct WrapperArrayStreamData* data =
      (struct WrapperArrayStreamData*)array_stream->private_data;
  array_stream->release = NULL;
  array_stream->private_data = NULL;

  // Export may fail before allocating the wrapper's private data.
  if (data == NULL) {
    return;
  }

  // Run the parent array stream release callback
  data->parent_array_stream->release(data->parent_array_stream);

  // If safe to do so, attempt to do an eager evaluation of a release
  // callback that may have been registered. If it is not safe to do so,
  // garbage collection will run any finalizers that have been set
  // on the chain of environments leading up to the finalizer.
  if (nanoarrow_is_main_thread()) {
    run_user_array_stream_finalizer(data->parent_array_stream_xptr);
  }

  nanoarrow_release_sexp(data->parent_array_stream_xptr);
  ArrowFree(data);
}

static const char* wrapper_array_stream_get_last_error(
    struct ArrowArrayStream* array_stream) {
  if (array_stream->release == NULL) {
    return "Array stream has been released";
  }
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
      nanoarrow_array_stream_from_xptr(parent_array_stream_xptr);

  // If there is no dependent object, don't bother with this wrapper
  SEXP dependent_sexp = R_ExternalPtrProtected(parent_array_stream_xptr);
  if (dependent_sexp == R_NilValue) {
    ArrowArrayStreamMove(parent_array_stream, array_stream_copy);
    return;
  }

  // Allocate a new external pointer for an array stream (for consistency:
  // we always move an array stream when exporting)
  SEXP parent_array_stream_xptr_new = PROTECT(nanoarrow_array_stream_owning_xptr());
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
