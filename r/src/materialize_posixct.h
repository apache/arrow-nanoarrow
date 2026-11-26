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

#ifndef R_MATERIALIZE_POSIXCT_H_INCLUDED
#define R_MATERIALIZE_POSIXCT_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "materialize_dbl.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_posixct(struct RConverter* converter) {
  if (converter->ptype_view.sexp_type != REALSXP) {
    return EINVAL;
  }

  if (converter->src.array_view->array->dictionary != NULL) {
    return ENOTSUP;
  }

  int64_t factor;
  switch (converter->schema_view.type) {
    case NANOARROW_TYPE_NA:
      return nanoarrow_materialize_dbl(converter);
    case NANOARROW_TYPE_DATE64:
      factor = 1000;
      break;
    case NANOARROW_TYPE_TIMESTAMP:
      switch (converter->schema_view.time_unit) {
        case NANOARROW_TIME_UNIT_SECOND:
          factor = 1;
          break;
        case NANOARROW_TIME_UNIT_MILLI:
          factor = 1000;
          break;
        case NANOARROW_TIME_UNIT_MICRO:
          factor = 1000000;
          break;
        case NANOARROW_TIME_UNIT_NANO:
          factor = 1000000000;
          break;
        default:
          return EINVAL;
      }
      break;
    default:
      return EINVAL;
  }

  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;
  if (src->array_view->storage_type != NANOARROW_TYPE_INT64) {
    return EINVAL;
  }

  double* result = REAL(dst->vec_sexp);
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;
  const int64_t* values = src->array_view->buffer_views[1].data.as_int64;
  int64_t n_bad_values = 0;

  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (is_valid != NULL && !ArrowBitGet(is_valid, raw_src_offset + i)) {
      result[dst->offset + i] = NA_REAL;
      continue;
    }

    int64_t value = values[raw_src_offset + i];
    int64_t whole_seconds = value / factor;
    int64_t subsecond = value % factor;
    if (whole_seconds > MAX_DBL_AS_INTEGER || whole_seconds < -MAX_DBL_AS_INTEGER) {
      n_bad_values++;
    }

    // Scaling before casting keeps lower-order ticks available for the
    // subsecond component instead of rounding the entire int64 value first.
    result[dst->offset + i] =
        (double)whole_seconds + ((double)subsecond / (double)factor);
  }

  if (n_bad_values > 0) {
    warn_lossy_conversion(
        n_bad_values,
        "may have incurred loss of whole-second precision in conversion to POSIXct()");
  }

  return NANOARROW_OK;
}

#endif
