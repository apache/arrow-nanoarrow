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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "util.h"

SEXP nanoarrow_ns_pkg = NULL;
SEXP nanoarrow_cls_array = NULL;
SEXP nanoarrow_cls_altrep_chr = NULL;
SEXP nanoarrow_cls_array_view = NULL;
SEXP nanoarrow_cls_data_frame = NULL;
SEXP nanoarrow_cls_schema = NULL;
SEXP nanoarrow_cls_array_stream = NULL;

void nanoarrow_init_cached_sexps(void) {
  SEXP nanoarrow_str = PROTECT(Rf_mkString("nanoarrow"));
  nanoarrow_ns_pkg = PROTECT(R_FindNamespace(nanoarrow_str));
  nanoarrow_cls_array = PROTECT(Rf_mkString("nanoarrow_array")); 
  nanoarrow_cls_altrep_chr = PROTECT(Rf_mkString("nanoarrow::altrep_chr")); 
  nanoarrow_cls_array_view = PROTECT(Rf_mkString("nanoarrow_array_view")); 
  nanoarrow_cls_data_frame = PROTECT(Rf_mkString("data.frame")); 
  nanoarrow_cls_schema = PROTECT(Rf_mkString("nanoarrow_schema")); 
  nanoarrow_cls_array_stream = PROTECT(Rf_mkString("nanoarrow_array_stream")); 

  R_PreserveObject(nanoarrow_ns_pkg);
  R_PreserveObject(nanoarrow_cls_array);
  R_PreserveObject(nanoarrow_cls_altrep_chr);
  R_PreserveObject(nanoarrow_cls_array_view);
  R_PreserveObject(nanoarrow_cls_data_frame);
  R_PreserveObject(nanoarrow_cls_schema);
  R_PreserveObject(nanoarrow_cls_array_stream);

  UNPROTECT(8);
}
