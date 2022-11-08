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

#ifndef R_MATERIALIZE_BLOB_H_INCLUDED
#define R_MATERIALIZE_BLOB_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "materialize_common.h"
#include "nanoarrow.h"

static inline int nanoarrow_materialize_blob(struct ArrayViewSlice* src,
                                             struct VectorSlice* dst,
                                             struct MaterializeOptions* options) {
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      break;
    default:
      return EINVAL;
  }

  if (src->array_view->storage_type == NANOARROW_TYPE_NA) {
    return NANOARROW_OK;
  }

  struct ArrowBufferView item;
  SEXP item_sexp;
  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
      item = ArrowArrayViewGetBytesUnsafe(src->array_view, src->offset + i);
      item_sexp = PROTECT(Rf_allocVector(RAWSXP, item.n_bytes));
      memcpy(RAW(item_sexp), item.data.data, item.n_bytes);
      SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i, item_sexp);
      UNPROTECT(1);
    }
  }

  return NANOARROW_OK;
}

#endif