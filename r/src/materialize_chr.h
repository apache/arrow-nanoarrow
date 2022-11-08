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

#include "materialize_common.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_chr(struct ArrayViewSlice* src,
                                            struct VectorSlice* dst,
                                            struct MaterializeOptions* options) {
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      break;
    default:
      return EINVAL;
  }

  if (src->array_view->storage_type == NANOARROW_TYPE_NA) {
    for (R_xlen_t i = 0; i < dst->length; i++) {
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
    }

    return NANOARROW_OK;
  }

  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(src->array_view, src->offset + i);
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i,
                     Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
    }
  }

  return NANOARROW_OK;
}

#endif
