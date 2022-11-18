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

SEXP nanoarrow_c_convert_array_stream(SEXP array_stream_xptr, SEXP ptype_sexp,
                                      SEXP size_sexp, SEXP n_sexp) {
  struct ArrowArrayStream* array_stream = array_stream_from_xptr(array_stream_xptr);
  double size = REAL(size_sexp)[0];
  double n = REAL(n_sexp)[0];

  SEXP schema_xptr = PROTECT(schema_owning_xptr());
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  int result = array_stream->get_schema(array_stream, schema);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStream::get_schema(): %s",
             array_stream->get_last_error(array_stream));
  }

  SEXP converter_xptr = PROTECT(nanoarrow_converter_from_ptype(ptype_sexp));
  if (nanoarrow_converter_set_schema(converter_xptr, schema_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  if (nanoarrow_converter_reserve(converter_xptr, size) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  SEXP array_xptr = PROTECT(array_owning_xptr());
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);

  int64_t n_batches = 0;
  int64_t n_materialized = 0;
  if (n > 0) {
    result = array_stream->get_next(array_stream, array);
    n_batches++;
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayStream::get_next(): %s",
               array_stream->get_last_error(array_stream));
    }

    while (array->release != NULL) {
      if (nanoarrow_converter_set_array(converter_xptr, array_xptr) != NANOARROW_OK) {
        nanoarrow_converter_stop(converter_xptr);
      }

      n_materialized = nanoarrow_converter_materialize_n(converter_xptr, array->length);
      if (n_materialized != array->length) {
        Rf_error("Expected to materialize %ld values in batch %ld but materialized %ld",
                 (long)array->length, (long)n_batches, (long)n_materialized);
      }

      if (n_batches >= n) {
        break;
      }

      result = array_stream->get_next(array_stream, array);
      n_batches++;
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayStream::get_next(): %s",
                 array_stream->get_last_error(array_stream));
      }
    }
  }

  if (nanoarrow_converter_finalize(converter_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  SEXP result_sexp = PROTECT(nanoarrow_converter_release_result(converter_xptr));
  UNPROTECT(4);
  return result_sexp;
}
