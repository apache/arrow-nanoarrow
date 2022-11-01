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

#ifndef R_MATERIALIZE_H_INCLUDED
#define R_MATERIALIZE_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

// These functions materialize a complete R vector or return R_NilValue
// if they cannot (i.e., no conversion possible). These functions will warn
// (once)  if there are values that cannot be converted (e.g., because they
// are out of range).
SEXP nanoarrow_materialize_unspecified(struct ArrowArrayView* array_view);
SEXP nanoarrow_materialize_lgl(struct ArrowArrayView* array_view);
SEXP nanoarrow_materialize_int(struct ArrowArrayView* array_view);
SEXP nanoarrow_materialize_dbl(struct ArrowArrayView* array_view);
SEXP nanoarrow_materialize_chr(struct ArrowArrayView* array_view);
SEXP nanoarrow_materialize_list_of_raw(struct ArrowArrayView* array_view);

#endif
