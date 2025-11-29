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

#ifndef R_NANOARROW_VCTR_BUILDER_LGL_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_LGL_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

class LglBuilder : public VctrBuilder {
 public:
  explicit LglBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_LGL, ptype_sexp) {}

  SEXP GetPtype() override { return Rf_allocVector(LGLSXP, 0); }

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::Reserve(n, error));
    SEXP value = PROTECT(Rf_allocVector(LGLSXP, n));
    SetValue(value);
    UNPROTECT(1);
    return NANOARROW_OK;
  }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(VctrBuilder::PushNext(array_shelter, array, error));

    // True for all the types supported here
    const uint8_t* is_valid = array_view_.buffer_views[0].data.as_uint8;
    const uint8_t* data_buffer = array_view_.buffer_views[1].data.as_uint8;

    int64_t raw_src_offset = array_view_.offset;
    R_xlen_t length = array->length;
    int* result = LOGICAL(value_);

    // Fill the buffer
    switch (array_view_.storage_type) {
      case NANOARROW_TYPE_NA:
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = NA_LOGICAL;
        }
        break;
      case NANOARROW_TYPE_BOOL:
        ArrowBitsUnpackInt32(data_buffer, raw_src_offset, length, result + value_size_);

        // Set any nulls to NA_LOGICAL
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_LOGICAL;
            }
          }
        }
        break;
      case NANOARROW_TYPE_INT8:
      case NANOARROW_TYPE_UINT8:
      case NANOARROW_TYPE_INT16:
      case NANOARROW_TYPE_UINT16:
      case NANOARROW_TYPE_INT32:
      case NANOARROW_TYPE_UINT32:
      case NANOARROW_TYPE_INT64:
      case NANOARROW_TYPE_UINT64:
      case NANOARROW_TYPE_FLOAT:
      case NANOARROW_TYPE_DOUBLE:
        for (R_xlen_t i = 0; i < length; i++) {
          result[value_size_ + i] = ArrowArrayViewGetIntUnsafe(&array_view_, i) != 0;
        }

        // Set any nulls to NA_LOGICAL
        if (is_valid != NULL && array_view_.null_count != 0) {
          for (R_xlen_t i = 0; i < length; i++) {
            if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
              result[value_size_ + i] = NA_LOGICAL;
            }
          }
        }
        break;

      default:
        return EINVAL;
    }

    return NANOARROW_OK;
  }
};

#endif
