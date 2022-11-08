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

#ifndef R_MATERIALIZE_UNSPECIFIED_H_INCLUDED
#define R_MATERIALIZE_UNSPECIFIED_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_unspecified(struct ArrayViewSlice* src,
                                                    struct VectorSlice* dst,
                                                    struct MaterializeOptions* options) {
  int* result = LOGICAL(dst->vec_sexp);

  int64_t total_offset = src->array_view->array->offset + src->offset;
  int64_t length = src->length;
  const uint8_t* bits = src->array_view->buffer_views[0].data.as_uint8;

  if (length == 0 || src->array_view->storage_type == NANOARROW_TYPE_NA ||
      ArrowBitCountSet(bits, total_offset, length) == 0) {
    // We can blindly set all the values to NA_LOGICAL without checking
    for (int64_t i = 0; i < length; i++) {
      result[dst->offset + i] = NA_LOGICAL;
    }
  } else {
    // Count non-null values and warn
    int64_t n_bad_values = 0;
    for (int64_t i = 0; i < length; i++) {
      n_bad_values += ArrowBitGet(bits, total_offset + i);
      result[dst->offset + i] = NA_LOGICAL;
    }

    if (n_bad_values > 0) {
      Rf_warning("%ld non-null value(s) set to NA", (long)n_bad_values);
    }
  }

  return NANOARROW_OK;
}

#endif