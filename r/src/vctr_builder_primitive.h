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

#ifndef R_NANOARROW_VCTR_BUILDER_PRIMITIVE_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_PRIMITIVE_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

class BlobBuilder : public VctrBuilder {
 public:
  explicit BlobBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_BLOB, ptype_sexp) {}
};

class DateBuilder : public VctrBuilder {
 public:
  explicit DateBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_DATE, ptype_sexp) {}
};

class HmsBuilder : public VctrBuilder {
 public:
  explicit HmsBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_HMS, ptype_sexp) {}
};

class PosixctBuilder : public VctrBuilder {
 public:
  explicit PosixctBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_POSIXCT, ptype_sexp) {}
};

class DifftimeBuilder : public VctrBuilder {
 public:
  explicit DifftimeBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_DIFFTIME, ptype_sexp) {}
};

class OtherBuilder : public VctrBuilder {
 public:
  explicit OtherBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_OTHER, ptype_sexp) {}
};

#endif
