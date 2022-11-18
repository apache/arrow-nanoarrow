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

#ifndef R_MATERIALIZE_LGL_H_INCLUDED
#define R_MATERIALIZE_LGL_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "nanoarrow.h"

static int nanoarrow_materialize_lgl(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                     struct MaterializeOptions* options) {
  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  const uint8_t* data_buffer = src->array_view->buffer_views[1].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;
  int* result = LOGICAL(dst->vec_sexp);

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_LOGICAL;
      }
      break;
    case NANOARROW_TYPE_BOOL:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = ArrowBitGet(data_buffer, src->offset + i);
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_LOGICAL;
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
      for (R_xlen_t i = 0; i < src->array_view->array->length; i++) {
        result[dst->offset + i] =
            ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i) != 0;
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_LOGICAL;
          }
        }
      }
      break;

    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

#endif
