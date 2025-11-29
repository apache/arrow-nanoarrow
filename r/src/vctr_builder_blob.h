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

#ifndef R_NANOARROW_VCTR_BUILDER_BLOB_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_BLOB_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

class BlobBuilder : public VctrBuilder {
 public:
  explicit BlobBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_BLOB, ptype_sexp) {}

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(VECSXP, n));
    SetValue(value);
    UNPROTECT(1);
    return NANOARROW_OK;
  }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::PushNext(array_shelter, array, error));
    R_xlen_t length = array_view_.length;

    switch (array_view_.storage_type) {
      case NANOARROW_TYPE_NA:
        // Works because lists are filled with R_NilValue by default
        // when allocated.
        return NANOARROW_OK;
      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_LARGE_STRING:
      case NANOARROW_TYPE_BINARY:
      case NANOARROW_TYPE_LARGE_BINARY:
        break;
      default:
        return ENOTSUP;
    }

    struct ArrowBufferView item;
    SEXP item_sexp;
    for (R_xlen_t i = 0; i < length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, i)) {
        item = ArrowArrayViewGetBytesUnsafe(&array_view_, i);
        item_sexp = PROTECT(Rf_allocVector(RAWSXP, item.size_bytes));
        memcpy(RAW(item_sexp), item.data.data, item.size_bytes);
        SET_VECTOR_ELT(value_, value_size_ + i, item_sexp);
        UNPROTECT(1);
      }
    }

    return NANOARROW_OK;
  }
};

#endif
