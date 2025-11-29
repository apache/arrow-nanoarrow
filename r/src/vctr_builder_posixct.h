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

#ifndef R_NANOARROW_VCTR_BUILDER_POSIXCT_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_POSIXCT_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_dbl.h"

class PosixctBuilder : public DblBuilder {
 public:
  explicit PosixctBuilder(SEXP ptype_sexp)
      : DblBuilder(ptype_sexp, VECTOR_TYPE_POSIXCT), scale_(0) {}

  ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                      ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(DblBuilder::Init(schema, options, error));

    ArrowTimeUnit time_unit = NANOARROW_TIME_UNIT_SECOND;
    switch (schema_view_.type) {
      case NANOARROW_TYPE_NA:
        break;
      case NANOARROW_TYPE_DATE64:
        time_unit = NANOARROW_TIME_UNIT_MILLI;
        break;
      case NANOARROW_TYPE_TIMESTAMP:
        time_unit = schema_view_.time_unit;
        break;
      default:
        StopCantConvert();
    }

    scale_ = 1;

    switch (time_unit) {
      case NANOARROW_TIME_UNIT_SECOND:
        scale_ *= 1;
        break;
      case NANOARROW_TIME_UNIT_MILLI:
        scale_ *= 1e-3;
        break;
      case NANOARROW_TIME_UNIT_MICRO:
        scale_ *= 1e-6;
        break;
      case NANOARROW_TIME_UNIT_NANO:
        scale_ *= 1e-9;
        break;
      default:
        return EINVAL;
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    R_xlen_t value_size0 = value_size_;
    NANOARROW_RETURN_NOT_OK(DblBuilder::PushNext(array_shelter, array, error));

    if (scale_ != 1) {
      double* result = REAL(value_);
      for (int64_t i = 0; i < array_view_.length; i++) {
        result[value_size0 + i] = result[value_size0 + i] * scale_;
      }
    }

    return NANOARROW_OK;
  }

 private:
  double scale_;
};

#endif
