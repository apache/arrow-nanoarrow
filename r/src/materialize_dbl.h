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

#ifndef R_MATERIALIZE_DBL_H_INCLUDED
#define R_MATERIALIZE_DBL_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "nanoarrow.h"
#include "util.h"

// Fall back to arrow for decimal conversion via a package helper
static inline void nanoarrow_materialize_decimal_to_dbl(struct RConverter* converter) {
  // A unique situation where we don't want owning external pointers because we know
  // these are protected for the duration of our call into R and because we don't want
  // then to be garbage collected and invalidate the converter
  SEXP array_xptr =
      PROTECT(R_MakeExternalPtr(converter->array_view.array, R_NilValue, R_NilValue));
  Rf_setAttrib(array_xptr, R_ClassSymbol, nanoarrow_cls_array);
  SEXP schema_xptr =
      PROTECT(R_MakeExternalPtr(converter->schema_view.schema, R_NilValue, R_NilValue));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);

  SEXP offset_sexp = PROTECT(Rf_ScalarReal(converter->src.offset));
  SEXP length_sexp = PROTECT(Rf_ScalarReal(converter->src.length));

  SEXP fun = PROTECT(Rf_install("convert_decimal_to_double"));
  SEXP call = PROTECT(Rf_lang5(fun, array_xptr, schema_xptr, offset_sexp, length_sexp));
  SEXP result_src = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));
  if (Rf_xlength(result_src) != converter->dst.length) {
    Rf_error("Unexpected result in call to Arrow for decimal conversion");
  }

  memcpy(REAL(converter->dst.vec_sexp) + converter->dst.offset, REAL(result_src),
         converter->dst.length * sizeof(double));
  UNPROTECT(7);
}

static inline int nanoarrow_materialize_dbl(struct RConverter* converter) {
  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;
  double* result = REAL(dst->vec_sexp);

  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_REAL;
      }
      break;
    case NANOARROW_TYPE_DOUBLE:
      memcpy(result + dst->offset,
             src->array_view->buffer_views[1].data.as_double + raw_src_offset,
             dst->length * sizeof(double));

      // Set any nulls to NA_REAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_REAL;
          }
        }
      }
      break;
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
      // TODO: implement bounds check for int64 and uint64, but instead
      // of setting to NA, just warn (because sequential values might not
      // roundtrip above 2^51 ish)
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] =
            ArrowArrayViewGetDoubleUnsafe(src->array_view, src->offset + i);
      }

      // Set any nulls to NA_REAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_REAL;
          }
        }
      }
      break;

    case NANOARROW_TYPE_DECIMAL128:
      nanoarrow_materialize_decimal_to_dbl(converter);
      break;

    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

#endif
