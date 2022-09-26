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

#ifndef R_NANOARROW_ARRAY_VIEW_H_INCLUDED
#define R_NANOARROW_ARRAY_VIEW_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

SEXP array_view_xptr_from_array_xptr(SEXP array_xptr);

static inline struct ArrowArrayView* array_view_from_xptr(SEXP array_view_xptr) {
  if (!Rf_inherits(array_view_xptr, "nanoarrow_array_view")) {
    Rf_error("`array_view` argument that is not a nanoarrow_array_view()");
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  if (array_view == NULL) {
    Rf_error("nanoarrow_array_view() is an external pointer to NULL");
  }

  return array_view;
}

#endif
