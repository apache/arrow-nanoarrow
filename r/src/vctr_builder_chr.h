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

#ifndef R_NANOARROW_VCTR_BUILDER_CHR_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_CHR_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <inttypes.h>
#include <stdint.h>

#include "vctr_builder_base.h"

class ChrBuilder : public VctrBuilder {
 public:
  explicit ChrBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_CHR, ptype_sexp) {}

  SEXP GetPtype() override { return Rf_allocVector(STRSXP, 0); }

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(STRSXP, n));
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
        for (R_xlen_t i = 0; i < length; i++) {
          SET_STRING_ELT(value_, value_size_ + i, NA_STRING);
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
        for (R_xlen_t i = 0; i < length; i++) {
          if (ArrowArrayViewIsNull(&array_view_, i)) {
            SET_STRING_ELT(value_, value_size_ + i, NA_STRING);
          } else {
            int n_chars = snprintf(buf, sizeof(buf), "%" PRId64,
                                   ArrowArrayViewGetIntUnsafe(&array_view_, i));
            SET_STRING_ELT(value_, value_size_ + i,
                           Rf_mkCharLenCE(buf, n_chars, CE_UTF8));
          }
        }
        return NANOARROW_OK;
      }

      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_LARGE_STRING: {
        struct ArrowStringView item;
        for (R_xlen_t i = 0; i < length; i++) {
          if (ArrowArrayViewIsNull(&array_view_, i)) {
            SET_STRING_ELT(value_, value_size_ + i, NA_STRING);
          } else {
            item = ArrowArrayViewGetStringUnsafe(&array_view_, i);
            SET_STRING_ELT(value_, value_size_ + i,
                           Rf_mkCharLenCE(item.data, (int)item.size_bytes, CE_UTF8));
          }
        }

        return NANOARROW_OK;
      }

      default:
        return ENOTSUP;
    }
  }
};

#endif
