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

#include "buffer.h"
#include "materialize_common.h"
#include "nanoarrow.h"

// bit64::as.integer64(2^53)
#define MAX_DBL_AS_INTEGER 9007199254740992

static inline int nanoarrow_decimal_to_chr(struct ArrowDecimal* item,
                                           struct ArrowBuffer* buffer, int scale);

static inline int nanoarrow_materialize_dbl(struct RConverter* converter) {
  if (converter->src.array_view->array->dictionary != NULL) {
    return ENOTSUP;
  }

  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;
  double* result = REAL(dst->vec_sexp);
  int64_t n_bad_values = 0;

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
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
      // No need to bounds check these types
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

    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        double value = ArrowArrayViewGetDoubleUnsafe(src->array_view, src->offset + i);
        if (value > MAX_DBL_AS_INTEGER || value < -MAX_DBL_AS_INTEGER) {
          // Content of null slot is undefined
          n_bad_values += is_valid == NULL || ArrowBitGet(is_valid, raw_src_offset + i);
        }

        result[dst->offset + i] = value;
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

    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256: {
      struct ArrowDecimal item;
      ArrowDecimalInit(&item, converter->schema_view.decimal_bitwidth,
                       converter->schema_view.decimal_precision,
                       converter->schema_view.decimal_scale);

      // Buffer to manage the building of the digits output
      SEXP buffer_xptr = PROTECT(buffer_owning_xptr());
      struct ArrowBuffer* digits = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);

      // A length one character() we'll use as input to Rf_asReal()
      SEXP decimal_as_chr = PROTECT(Rf_allocVector(STRSXP, 1));

      for (R_xlen_t i = 0; i < dst->length; i++) {
        if (is_valid != NULL && !ArrowBitGet(is_valid, raw_src_offset + i)) {
          result[dst->offset + i] = NA_REAL;
          continue;
        }

        ArrowArrayViewGetDecimalUnsafe(src->array_view, src->offset + i, &item);
        digits->size_bytes = 0;
        int status = ArrowDecimalAppendStringToBuffer(&item, digits);
        if (status != NANOARROW_OK) {
          UNPROTECT(2);
          return status;
        }

        SET_STRING_ELT(decimal_as_chr, 0,
                       Rf_mkCharLen((char*)digits->data, (int)digits->size_bytes));
        result[dst->offset + i] = Rf_asReal(decimal_as_chr);
      }
      UNPROTECT(2);
      return NANOARROW_OK;
    }

    default:
      return EINVAL;
  }

  if (n_bad_values > 0) {
    warn_lossy_conversion(
        n_bad_values, "may have incurred loss of precision in conversion to double()");
  }

  return NANOARROW_OK;
}

#endif
