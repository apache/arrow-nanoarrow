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

#ifndef R_NANOARROW_SCHEMA_H_INCLUDED
#define R_NANOARROW_SCHEMA_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include <nanoarrow/r.h>
#include "nanoarrow.h"
#include "util.h"

// Returns an external pointer to a schema child. The returned pointer will keep its
// parent alive: this is typically what you want when printing or performing a conversion,
// where the borrowed external pointer is ephemeral.
SEXP borrow_schema_child_xptr(SEXP schema_xptr, int64_t i);

// Returns the underlying struct ArrowSchema* from an external pointer,
// checking and erroring for invalid objects, pointers, and arrays, but
// allowing for R_NilValue to signify a NULL return.
static inline struct ArrowSchema* nullable_schema_from_xptr(SEXP schema_xptr) {
  if (schema_xptr == R_NilValue) {
    return NULL;
  } else {
    return nanoarrow_schema_from_xptr(schema_xptr);
  }
}

static inline void schema_export(SEXP schema_xptr, struct ArrowSchema* schema_copy) {
  int result = ArrowSchemaDeepCopy(nanoarrow_schema_from_xptr(schema_xptr), schema_copy);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaDeepCopy() failed");
  }
}

#endif
