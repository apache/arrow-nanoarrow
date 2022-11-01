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

// Note: These conversions are not currently written for safety rather than
// speed. We could make use of C++ templating to provide faster and/or more
// readable conversions here with a C entry point.

SEXP nanoarrow_materialize_unspecified(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(LGLSXP, array_view->array->length));
  Rf_setAttrib(result_sexp, R_ClassSymbol, Rf_mkString("vctrs_unspecified"));
  int* result = LOGICAL(result_sexp);

  int64_t length = array_view->array->length;
  const uint8_t* bits = array_view->buffer_views[0].data.as_uint8;
  int64_t null_count = array_view->array->null_count;

  if (length > 0 && null_count == -1 && bits != NULL &&
      array_view->layout.buffer_type[0] == NANOARROW_BUFFER_TYPE_VALIDITY) {
    null_count = length - ArrowBitCountSet(bits, array_view->array->offset, length);
  }

  if (length == 0 || length == null_count ||
      array_view->storage_type == NANOARROW_TYPE_NA) {
    // We can blindly set all the values to NA_LOGICAL without checking
    for (int64_t i = 0; i < length; i++) {
      result[i] = NA_LOGICAL;
    }
  } else {
    // Count non-null values and warn
    int64_t n_bad_values = 0;
    for (int64_t i = 0; i < length; i++) {
      n_bad_values += ArrowBitGet(bits, array_view->array->offset + i);
      result[i] = NA_LOGICAL;
    }

    if (n_bad_values > 0) {
      Rf_warning("%ld non-null value(s) set to NA", (long)n_bad_values);
    }
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_lgl(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(LGLSXP, array_view->array->length));
  int* result = LOGICAL(result_sexp);

  // True for all the types supported here
  const uint8_t* is_valid = array_view->buffer_views[0].data.as_uint8;
  const uint8_t* data_buffer = array_view->buffer_views[1].data.as_uint8;

  // Fill the buffer
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_BOOL:
      for (R_xlen_t i = 0; i < array_view->array->length; i++) {
        result[i] = ArrowBitGet(data_buffer, i);
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_LOGICAL;
          }
        }
      }
      break;
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      for (R_xlen_t i = 0; i < array_view->array->length; i++) {
        result[i] = ArrowArrayViewGetIntUnsafe(array_view, i) != 0;
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < array_view->array->length; i++) {
          if (!ArrowBitGet(is_valid, i)) {
            result[i] = NA_LOGICAL;
          }
        }
      }
      break;

    default:
      UNPROTECT(1);
      return R_NilValue;
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
      UNPROTECT(1);
      return R_NilValue;
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
      UNPROTECT(1);
      return R_NilValue;
  }

  UNPROTECT(1);
  return result_sexp;
}

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

SEXP nanoarrow_materialize_list_of_raw(struct ArrowArrayView* array_view) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      break;
    default:
      return R_NilValue;
  }

  SEXP result_sexp = PROTECT(Rf_allocVector(VECSXP, array_view->array->length));

  struct ArrowBufferView item;
  SEXP item_sexp;
  for (R_xlen_t i = 0; i < array_view->array->length; i++) {
    if (!ArrowArrayViewIsNull(array_view, i)) {
      item = ArrowArrayViewGetBytesUnsafe(array_view, i);
      item_sexp = PROTECT(Rf_allocVector(RAWSXP, item.n_bytes));
      memcpy(RAW(item_sexp), item.data.data, item.n_bytes);
      SET_VECTOR_ELT(result_sexp, i, item_sexp);
      UNPROTECT(1);
    }
  }

  UNPROTECT(1);
  return result_sexp;
}
