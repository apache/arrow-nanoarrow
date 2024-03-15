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

#ifndef R_NANOARROW_VCTR_BUILDER_UNSPECIFIED_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_UNSPECIFIED_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

class UnspecifiedBuilder : public VctrBuilder {
 public:
  explicit UnspecifiedBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_UNSPECIFIED, ptype_sexp) {}

  ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                      ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Init(schema, options, error));
    switch (schema_view_.type) {
      case NANOARROW_TYPE_DICTIONARY:
        StopCantConvert();
      default:
        break;
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(LGLSXP, n));
    SetValue(value);
    UNPROTECT(1);
    return NANOARROW_OK;
  }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    int64_t not_null_count;
    if (array->null_count == -1 && array->buffers[0] == nullptr) {
      not_null_count = array->length;
    } else if (array->null_count == -1) {
      not_null_count =
          ArrowBitCountSet(reinterpret_cast<const uint8_t*>(array->buffers[0]),
                           array->offset, array->length);
    } else {
      not_null_count = array->length - array->null_count;
    }

    if (not_null_count > 0 && array->length > 0) {
      NANOARROW_RETURN_NOT_OK(
          WarnLossyConvert("that were non-null set to NA", not_null_count));
    }

    int* value_ptr = LOGICAL(value_) + value_size_;
    for (int64_t i = 0; i < array->length; i++) {
      value_ptr[i] = NA_LOGICAL;
    }

    return NANOARROW_OK;
  }
};

#endif
