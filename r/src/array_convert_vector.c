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

#include "nanoarrow.h"

#include "array.h"

enum VectorType {
  VECTOR_TYPE_LOGICAL,
  VECTOR_TYPE_INTEGER,
  VECTOR_TYPE_DOUBLE,
  VECTOR_TYPE_CHARACTER,
  VECTOR_TYPE_DATA_FRAME,
  VECTOR_TYPE_LIST_OF_RAW,
  VECTOR_TYPE_OTHER
};

// These conversions are the conversion we can be sure about without inspecting
// any extra data from schema or the array.
static enum VectorType vector_type_from_array_type(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_BOOL:
      return VECTOR_TYPE_LOGICAL;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
      return VECTOR_TYPE_INTEGER;

    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      return VECTOR_TYPE_DOUBLE;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      return VECTOR_TYPE_CHARACTER;

    case NANOARROW_TYPE_STRUCT:
      return VECTOR_TYPE_DATA_FRAME;

    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      return VECTOR_TYPE_LIST_OF_RAW;

    default:
      break;
  }
}

static enum VectorType vector_type_from_array_view_xptr(SEXP array_view_xptr) {
  SEXP array_xptr = R_ExternalPtrProtected(array_view_xptr);
  struct ArrowArray* array = array_from_xptr(array_xptr);
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);
}

static SEXP ptype_from_array_type(enum ArrowType type) {
  SEXP ptype = R_NilValue;
  switch (type) {
    case NANOARROW_TYPE_STRUCT:
      ptype = PROTECT(Rf_allocVector(VECSXP, 0));
      Rf_setAttrib(ptype, R_ClassSymbol, Rf_mkString("data.frame"));
      UNPROTECT(1);
      break;
    case NANOARROW_TYPE_STRING:
      return Rf_allocVector(STRSXP, 0);
    default:
      break;
  }

  return ptype;
}
