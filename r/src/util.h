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

#ifndef R_UTIL_H_INCLUDED
#define R_UTIL_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

extern SEXP nanoarrow_ns_pkg;
extern SEXP nanoarrow_cls_array;
extern SEXP nanoarrow_cls_altrep_chr;
extern SEXP nanoarrow_cls_array_view;
extern SEXP nanoarrow_cls_data_frame;
extern SEXP nanoarrow_cls_schema;
extern SEXP nanoarrow_cls_array_stream;

void nanoarrow_init_cached_sexps(void);

#endif
