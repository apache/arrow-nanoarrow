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

#ifndef R_MATERIALIZE_INT_H_INCLUDED
#define R_MATERIALIZE_INT_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_int(struct ArrayViewSlice* src,
                                            struct VectorSlice* dst,
                                            struct MaterializeOptions* options) {
  int* result = INTEGER(dst->vec_sexp);
  int64_t n_bad_values = 0;

  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_INTEGER;
      }
      break;
    case NANOARROW_TYPE_INT32:
      memcpy(result + dst->offset,
             src->array_view->buffer_views[1].data.as_int32 + raw_src_offset,
             dst->length * sizeof(int32_t));

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_INTEGER;
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
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] =
            ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
      }

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_INTEGER;
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
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (ArrowBitGet(is_valid, raw_src_offset + i)) {
            int64_t value = ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
            if (value > INT_MAX || value <= NA_INTEGER) {
              result[dst->offset + i] = NA_INTEGER;
              n_bad_values++;
            } else {
              result[dst->offset + i] = value;
            }
          } else {
            result[dst->offset + i] = NA_INTEGER;
          }
        }
      } else {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          int64_t value = ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
          if (value > INT_MAX || value <= NA_INTEGER) {
            result[dst->offset + i] = NA_INTEGER;
            n_bad_values++;
          } else {
            result[dst->offset + i] = value;
          }
        }
      }
      break;

    default:
      return EINVAL;
  }

  if (n_bad_values > 0) {
    Rf_warning("%ld value(s) outside integer range set to NA", (long)n_bad_values);
  }

  return NANOARROW_OK;
}

#endif
