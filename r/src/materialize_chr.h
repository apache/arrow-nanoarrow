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

#ifndef R_MATERIALIZE_CHR_H_INCLUDED
#define R_MATERIALIZE_CHR_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include <inttypes.h>

#include "buffer.h"
#include "materialize_common.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_chr(struct RConverter* converter) {
  if (converter->src.array_view->array->dictionary != NULL) {
    return ENOTSUP;
  }

  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;

  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
      }
      return NANOARROW_OK;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64: {
      char buf[64];
      for (R_xlen_t i = 0; i < dst->length; i++) {
        if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
        } else {
          int n_chars =
              snprintf(buf, sizeof(buf), "%" PRId64,
                       ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i));
          SET_STRING_ELT(dst->vec_sexp, dst->offset + i,
                         Rf_mkCharLenCE(buf, n_chars, CE_UTF8));
        }
      }
      return NANOARROW_OK;
    }

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

      for (R_xlen_t i = 0; i < dst->length; i++) {
        if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
        } else {
          ArrowArrayViewGetDecimalUnsafe(src->array_view, src->offset + i, &item);
          digits->size_bytes = 0;
          int status = ArrowDecimalAppendStringToBuffer(&item, digits);
          if (status != NANOARROW_OK) {
            UNPROTECT(1);
            return status;
          }

          SET_STRING_ELT(dst->vec_sexp, dst->offset + i,
                         Rf_mkCharLen((char*)digits->data, (int)digits->size_bytes));
        }
      }
      UNPROTECT(1);
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_STRING_VIEW:
      break;

    default:
      return ENOTSUP;
  }

  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(src->array_view, src->offset + i);
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i,
                     Rf_mkCharLenCE(item.data, (int)item.size_bytes, CE_UTF8));
    }
  }

  return NANOARROW_OK;
}

#endif
