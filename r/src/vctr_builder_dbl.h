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

#ifndef R_NANOARROW_VCTR_BUILDER_DBL_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_DBL_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

// bit64::as.integer64(2^53)
#define MAX_DBL_AS_INTEGER 9007199254740992

class DblBuilder : public VctrBuilder {
 public:
  explicit DblBuilder(SEXP ptype_sexp, VectorType vector_type = VECTOR_TYPE_DBL)
      : VctrBuilder(vector_type, ptype_sexp) {}

  SEXP GetPtype() override {
    if (ptype_sexp_ != R_NilValue) {
      return ptype_sexp_;
    } else {
      return Rf_allocVector(REALSXP, 0);
    }
  }

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(REALSXP, n));
    SetValue(value);
    UNPROTECT(1);
    return NANOARROW_OK;
  }

  virtual ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                                  ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::PushNext(array_shelter, array, error));

    double* result = REAL(value_);
    int64_t n_bad_values = 0;

    // True for all the types supported here
    const uint8_t* is_valid = array_view_.buffer_views[0].data.as_uint8;
    int64_t raw_src_offset = array_view_.offset;
    R_xlen_t length = array_view_.length;

    // Fill the buffer
    switch (array_view_.storage_type) {
      case NANOARROW_TYPE_NA:
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = NA_REAL;
        }
        break;
      case NANOARROW_TYPE_DOUBLE:
        memcpy(result + value_size_,
               array_view_.buffer_views[1].data.as_double + raw_src_offset,
               length * sizeof(double));

        // Set any nulls to NA_REAL
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_REAL;
            }
          }
        }
        break;
      case NANOARROW_TYPE_BOOL:
      case NANOARROW_TYPE_INT8:
      case NANOARROW_TYPE_UINT8:
      case NANOARROW_TYPE_INT16:
      case NANOARROW_TYPE_UINT16:
      case NANOARROW_TYPE_INT32:
      case NANOARROW_TYPE_UINT32:
      case NANOARROW_TYPE_FLOAT:
        // No need to bounds check these types
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = ArrowArrayViewGetDoubleUnsafe(&array_view_, i);
        }

        // Set any nulls to NA_REAL
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_REAL;
            }
          }
        }
        break;

      case NANOARROW_TYPE_INT64:
      case NANOARROW_TYPE_UINT64:
        for (R_xlen_t i = 0; i < length; i++) {
          double value = ArrowArrayViewGetDoubleUnsafe(&array_view_, i);
          if (value > MAX_DBL_AS_INTEGER || value < -MAX_DBL_AS_INTEGER) {
            // Content of null slot is undefined
            n_bad_values += is_valid == NULL || ArrowBitGet(is_valid, raw_src_offset + i);
          }

          result[value_size_ + i] = value;
        }

        // Set any nulls to NA_REAL
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_REAL;
            }
          }
        }
        break;

      default:
        return EINVAL;
    }

    if (n_bad_values > 0) {
      WarnLossyConvert("may have incurred loss of precision in conversion to double()",
                       n_bad_values);
    }

    return NANOARROW_OK;
  }
};

#endif
