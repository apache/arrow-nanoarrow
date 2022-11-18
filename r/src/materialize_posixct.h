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
  if (converter->ptype_view.sexp_type == REALSXP) {
    enum ArrowTimeUnit time_unit;
    switch (converter->schema_view.data_type) {
      case NANOARROW_TYPE_NA:
        time_unit = NANOARROW_TIME_UNIT_SECOND;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(converter));
        break;
      case NANOARROW_TYPE_DATE64:
        time_unit = NANOARROW_TIME_UNIT_MILLI;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(converter));
        break;
      case NANOARROW_TYPE_TIMESTAMP:
        time_unit = converter->schema_view.time_unit;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(converter));
        break;
      default:
        return EINVAL;
    }

    double scale;
    switch (time_unit) {
      case NANOARROW_TIME_UNIT_SECOND:
        scale = 1;
        break;
      case NANOARROW_TIME_UNIT_MILLI:
        scale = 1e-3;
        break;
      case NANOARROW_TIME_UNIT_MICRO:
        scale = 1e-6;
        break;
      case NANOARROW_TIME_UNIT_NANO:
        scale = 1e-9;
        break;
      default:
        return EINVAL;
    }

    if (scale != 1) {
      double* result = REAL(converter->dst.vec_sexp);
      for (int64_t i = 0; i < converter->dst.length; i++) {
        result[converter->dst.offset + i] = result[converter->dst.offset + i] * scale;
      }
    }

    return NANOARROW_OK;
  }

  return EINVAL;
}

#endif
