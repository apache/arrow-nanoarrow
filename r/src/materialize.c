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

SEXP nanoarrow_materialize_chr(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(STRSXP, array_view->array->length));

  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < array_view->array->length; i++) {
    if (ArrowArrayViewIsNull(array_view, i)) {
      SET_STRING_ELT(result_sexp, i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(array_view, i);
      SET_STRING_ELT(result_sexp, i, Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
    }
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_int(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(INTSXP, array_view->array->length));
  int* result = INTEGER(result_sexp);
  int64_t n_bad_values = 0;

  // True for all the types supported here
  const uint8_t* is_valid = array_view->buffer_views[0].data.as_uint8;

  // Fill the buffer
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT32:
      memcpy(result,
             array_view->buffer_views[1].data.as_int32 + array_view->array->offset,
             array_view->array->length * sizeof(int32_t));

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_INTEGER;
          }
        }
      }
      break;
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
      // No need to bounds check for these types
      for (R_xlen_t i = 0; i < array_view->array->length; i++) {
        result[i] = ArrowArrayViewGetIntUnsafe(array_view, i);
      }

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_INTEGER;
          }
        }
      }
      break;
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      // Loop + bounds check. Because we don't know what memory might be
      // in a null slot, we have to check nulls if there are any.
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (ArrowBitGet(is_valid, i)) {
            int64_t value = ArrowArrayViewGetIntUnsafe(array_view, i);
            if (value > INT_MAX || value <= NA_INTEGER) {
              result[i] = NA_INTEGER;
              n_bad_values++;
            } else {
              result[i] = value;
            }
          } else {
            result[i] = NA_INTEGER;
          }
        }
      } else {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          int64_t value = ArrowArrayViewGetIntUnsafe(array_view, i);
          if (value > INT_MAX || value <= NA_INTEGER) {
            result[i] = NA_INTEGER;
            n_bad_values++;
          } else {
            result[i] = value;
          }
        }
      }
      break;

    default:
      Rf_error("Can't convert array to integer()");
  }

  if (n_bad_values > 0) {
    Rf_warning("%ld value(s) outside integer range set to NA", (long)n_bad_values);
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_dbl(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(REALSXP, array_view->array->length));
  double* result = REAL(result_sexp);

  // True for all the types supported here
  const uint8_t* is_valid = array_view->buffer_views[0].data.as_uint8;

  // Fill the buffer
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DOUBLE:
      memcpy(result,
             array_view->buffer_views[1].data.as_double + array_view->array->offset,
             array_view->array->length * sizeof(double));

      // Set any nulls to NA_REAL
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_REAL;
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
      for (R_xlen_t i = 0; i < array_view->array->length; i++) {
        result[i] = ArrowArrayViewGetDoubleUnsafe(array_view, i);
      }

      // Set any nulls to NA_REAL
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_REAL;
          }
        }
      }
      break;

    default:
      Rf_error("Can't convert array to double()");
  }

  UNPROTECT(1);
  return result_sexp;
}
