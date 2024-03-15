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

#ifndef R_NANOARROW_VCTR_BUILDER_INT64_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_INT64_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

#define NA_INTEGER64 INT64_MIN

class Integer64Builder : public VctrBuilder {
 public:
  explicit Integer64Builder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_INTEGER64, ptype_sexp) {}

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(REALSXP, n));
    SetValue(value);
    UNPROTECT(1);
    return NANOARROW_OK;
  }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::PushNext(array_shelter, array, error));

    int64_t* result = reinterpret_cast<int64_t*>(REAL(value_));
    int64_t n_bad_values = 0;

    // True for all the types supported here
    const uint8_t* is_valid = array_view_.buffer_views[0].data.as_uint8;
    int64_t raw_src_offset = array_view_.offset;
    R_xlen_t length = array->length;

    // Fill the buffer
    switch (array_view_.storage_type) {
      case NANOARROW_TYPE_NA:
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = NA_INTEGER64;
        }
        break;
      case NANOARROW_TYPE_INT64:
        memcpy(result + value_size_,
               array_view_.buffer_views[1].data.as_int32 + raw_src_offset,
               length * sizeof(int64_t));

        // Set any nulls to NA_INTEGER64
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_INTEGER64;
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
        // No need to bounds check for these types
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = ArrowArrayViewGetIntUnsafe(&array_view_, i);
        }

        // Set any nulls to NA_INTEGER
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_INTEGER64;
            }
          }
        }
        break;
      case NANOARROW_TYPE_UINT64:
      case NANOARROW_TYPE_FLOAT:
      case NANOARROW_TYPE_DOUBLE:
        // Loop + bounds check. Because we don't know what memory might be
        // in a null slot, we have to check nulls if there are any.
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (ArrowBitGet(is_valid, raw_src_offset + i)) {
              int64_t value = ArrowArrayViewGetIntUnsafe(&array_view_, i);
              if (value > INT64_MAX || value <= NA_INTEGER64) {
                result[value_size_ + i] = NA_INTEGER64;
                n_bad_values++;
              } else {
                result[value_size_ + i] = value;
              }
            } else {
              result[value_size_ + i] = NA_INTEGER64;
            }
          }
        } else {
          for (R_xlen_t i = 0; i < length; i++) {
            int64_t value = ArrowArrayViewGetIntUnsafe(&array_view_, i);
            if (value > INT64_MAX || value <= NA_INTEGER64) {
              result[value_size_ + i] = NA_INTEGER64;
              n_bad_values++;
            } else {
              result[value_size_ + i] = value;
            }
          }
        }
        break;

      default:
        return EINVAL;
    }

    if (n_bad_values > 0) {
      WarnLossyConvert("outside integer64 range set to NA", n_bad_values);
    }

    return NANOARROW_OK;
  }
};

#endif
