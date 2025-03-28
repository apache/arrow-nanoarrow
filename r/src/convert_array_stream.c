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
#include "array_stream.h"
#include "convert.h"
#include "schema.h"

static int convert_next(SEXP converter_xptr, struct ArrowArrayStream* stream,
                        SEXP schema_xptr, int64_t* n_batches) {
  SEXP array_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* array = nanoarrow_output_array_from_xptr(array_xptr);

  // Fetch the next array
  int result = ArrowArrayStreamGetNext(stream, array, NULL);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStream::get_next(): %s", ArrowArrayStreamGetLastError(stream));
  }

  // Check if the stream is finished
  if (array->release == NULL) {
    UNPROTECT(1);
    return 0;
  }

  // Bump the batch counter
  (*n_batches)++;

  // Set the schema of the allocated array and pass it to the converter
  R_SetExternalPtrTag(array_xptr, schema_xptr);
  if (nanoarrow_converter_set_array(converter_xptr, array_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  // After set_array, the converter is responsible for the array_xptr
  UNPROTECT(1);

  // Materialize the array into the converter
  int64_t n_materialized =
      nanoarrow_converter_materialize_n(converter_xptr, array->length);
  if (n_materialized != array->length) {
    Rf_error("Expected to materialize %ld values in batch %ld but materialized %ld",
             (long)array->length, (long)(*n_batches), (long)n_materialized);
  }

  return 1;
}

SEXP nanoarrow_c_convert_array_stream(SEXP array_stream_xptr, SEXP ptype_sexp,
                                      SEXP size_sexp, SEXP n_sexp) {
  struct ArrowArrayStream* array_stream =
      nanoarrow_array_stream_from_xptr(array_stream_xptr);
  int64_t size = (int64_t)(REAL(size_sexp)[0]);

  double n_real = REAL(n_sexp)[0];
  int n;
  if (R_FINITE(n_real)) {
    n = (int)n_real;
  } else {
    n = INT_MAX;
  }

  SEXP schema_xptr = PROTECT(nanoarrow_schema_owning_xptr());
  struct ArrowSchema* schema = nanoarrow_output_schema_from_xptr(schema_xptr);

  int result = ArrowArrayStreamGetSchema(array_stream, schema, NULL);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStream::get_schema(): %s",
             ArrowArrayStreamGetLastError(array_stream));
  }

  SEXP converter_xptr = PROTECT(nanoarrow_converter_from_ptype(ptype_sexp));
  if (nanoarrow_converter_set_schema(converter_xptr, schema_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  nanoarrow_converter_reserve(converter_xptr, size);

  int64_t n_batches = 0;
  do {
    if (n_batches >= n) {
      break;
    }
  } while (convert_next(converter_xptr, array_stream, schema_xptr, &n_batches));

  if (nanoarrow_converter_finalize(converter_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  SEXP result_sexp = PROTECT(nanoarrow_converter_release_result(converter_xptr));
  UNPROTECT(3);
  return result_sexp;
}
