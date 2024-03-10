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

#ifndef R_NANOARROW_VCTR_BUILDER_HMS_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_HMS_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_difftime.h"

class HmsBuilder : public DifftimeBuilder {
 public:
  explicit HmsBuilder(SEXP ptype_sexp) : DifftimeBuilder(ptype_sexp, VECTOR_TYPE_HMS) {}

  ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                      ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(DifftimeBuilder::Init(schema, options, error));
    switch (schema_view_.type) {
      case NANOARROW_TYPE_NA:
      case NANOARROW_TYPE_TIME32:
      case NANOARROW_TYPE_TIME64:
        break;
      default:
        StopCantConvert();
    }

    return NANOARROW_OK;
  }
};

#endif
