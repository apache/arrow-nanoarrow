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

// Creates an external pointer to a struct ArrowArrayView, erroring
// if the validation inherent in its creation fails (i.e., calling
// this will aslo validate the array). This requires that array_xptr
// has a schema attached. The ArrowArrayView is an augmented structure
// provided by the nanoarrow C library that makes it easier to access
// elements and buffers. This is not currently exposed at the R
// level but is used at the C level to make validation and conversion
// to R easier to write.
SEXP array_view_xptr_from_array_xptr(SEXP array_xptr);

// Returns the struct ArrowArrayView underlying an external pointer,
// erroring for invalid objects and NULL pointers.
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
