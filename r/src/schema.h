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

#include "nanoarrow.h"

void finalize_schema_xptr(SEXP schema_xptr);

static inline struct ArrowSchema* schema_from_xptr(SEXP schema_xptr) {
  if (!Rf_inherits(schema_xptr, "nanoarrow_schema")) {
    Rf_error("`schema` argument that is not");
  }

  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema == NULL) {
    Rf_error("nanoarrow_schema() is an external pointer to NULL");
  }

  if (schema->release == NULL) {
    Rf_error("nanoarrow_schema() has already been released");
  }

  return schema;
}

static inline struct ArrowSchema* nullable_schema_from_xptr(SEXP schema_xptr) {
  if (schema_xptr == R_NilValue) {
    return NULL;
  } else {
    return schema_from_xptr(schema_xptr);
  }
}

static inline SEXP schema_owning_xptr() {
  struct ArrowSchema* schema =
      (struct ArrowSchema*)ArrowMalloc(sizeof(struct ArrowSchema));
  schema->release = NULL;

  SEXP schema_xptr = PROTECT(R_MakeExternalPtr(schema, R_NilValue, R_NilValue));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, Rf_mkString("nanoarrow_schema"));
  R_RegisterCFinalizer(schema_xptr, &finalize_schema_xptr);
  UNPROTECT(1);
  return schema_xptr;
}

#endif
