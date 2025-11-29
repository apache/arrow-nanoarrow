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

#ifndef R_NANOARROW_VCTR_BUILDER_DIFFTIME_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_DIFFTIME_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_dbl.h"

class DifftimeBuilder : public DblBuilder {
 public:
  explicit DifftimeBuilder(SEXP ptype_sexp, VectorType vector_type = VECTOR_TYPE_DIFFTIME)
      : DblBuilder(ptype_sexp, vector_type), scale_(0) {}

  ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                      ArrowError* error) override {
    NANOARROW_RETURN_NOT_OK(DblBuilder::Init(schema, options, error));
    switch (schema_view_.type) {
      case NANOARROW_TYPE_NA:
      case NANOARROW_TYPE_TIME32:
      case NANOARROW_TYPE_TIME64:
      case NANOARROW_TYPE_DURATION:
        break;
      default:
        StopCantConvert();
    }

    switch (GetTimeUnits(ptype_sexp_)) {
      case R_TIME_UNIT_MINUTES:
        scale_ = 1.0 / 60;
        break;
      case R_TIME_UNIT_HOURS:
        scale_ = 1.0 / (60 * 60);
        break;
      case R_TIME_UNIT_DAYS:
        scale_ = 1.0 / (60 * 60 * 24);
        break;
      case R_TIME_UNIT_WEEKS:
        scale_ = 1.0 / (60 * 60 * 24 * 7);
        break;
      default:
        scale_ = 1.0;
        break;
    }

    switch (schema_view_.time_unit) {
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

  static RTimeUnits GetTimeUnits(SEXP ptype) {
    SEXP units_attr = Rf_getAttrib(ptype, Rf_install("units"));
    if (units_attr == R_NilValue || TYPEOF(units_attr) != STRSXP ||
        Rf_length(units_attr) != 1) {
      Rf_error("Expected difftime 'units' attribute of type character(1)");
    }

    const char* dst_units = Rf_translateCharUTF8(STRING_ELT(units_attr, 0));
    if (strcmp(dst_units, "secs") == 0) {
      return R_TIME_UNIT_SECONDS;
    } else if (strcmp(dst_units, "mins") == 0) {
      return R_TIME_UNIT_MINUTES;
    } else if (strcmp(dst_units, "hours") == 0) {
      return R_TIME_UNIT_HOURS;
    } else if (strcmp(dst_units, "days") == 0) {
      return R_TIME_UNIT_DAYS;
    } else if (strcmp(dst_units, "weeks") == 0) {
      return R_TIME_UNIT_WEEKS;
    } else {
      Rf_error("Unexpected value for difftime 'units' attribute");
      return R_TIME_UNIT_SECONDS;
    }
  }
};

#endif
