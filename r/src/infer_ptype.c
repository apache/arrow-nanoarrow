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

#include "altrep.h"
#include "array.h"
#include "array_view.h"
#include "materialize.h"

// These conversions are the default R-native type guesses for
// an array that don't require extra information from the ptype (e.g.,
// factor with levels). Some of these guesses may result in a conversion
// that later warns for out-of-range values (e.g., int64 to double());
// however, a user can use the materialize_array(x, ptype = something_safer())
// when this occurs.
enum VectorType nanoarrow_infer_vector_type(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_NA:
      return VECTOR_TYPE_UNSPECIFIED;

    case NANOARROW_TYPE_BOOL:
      return VECTOR_TYPE_LGL;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
      return VECTOR_TYPE_INT;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      return VECTOR_TYPE_DBL;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      return VECTOR_TYPE_CHR;

    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      return VECTOR_TYPE_LIST_OF_RAW;

    case NANOARROW_TYPE_STRUCT:
      return VECTOR_TYPE_DATA_FRAME;

    default:
      return VECTOR_TYPE_OTHER;
  }
}

// The same as the above, but from a nanoarrow_array()
enum VectorType nanoarrow_infer_vector_type_array(SEXP array_xptr) {
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);

  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  if (ArrowSchemaViewInit(&schema_view, schema, &error) != NANOARROW_OK) {
    Rf_error("nanoarrow_infer_vector_type_array(): %s", ArrowErrorMessage(&error));
  }

  return nanoarrow_infer_vector_type(schema_view.data_type);
}

// Call nanoarrow::infer_ptype_other(), which handles less common types that
// are easier to compute in R or gives an informative error if this is
// not possible.
static SEXP call_infer_ptype_other(SEXP array_xptr) {
  SEXP ns = PROTECT(R_FindNamespace(Rf_mkString("nanoarrow")));
  SEXP call = PROTECT(Rf_lang2(Rf_install("infer_ptype_other"), array_xptr));
  SEXP result = PROTECT(Rf_eval(call, ns));
  UNPROTECT(3);
  return result;
}

SEXP nanoarrow_c_infer_ptype(SEXP array_xptr);

static SEXP infer_ptype_data_frame(SEXP array_xptr) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  SEXP result = PROTECT(Rf_allocVector(VECSXP, array->n_children));
  SEXP result_names = PROTECT(Rf_allocVector(STRSXP, array->n_children));

  for (R_xlen_t i = 0; i < array->n_children; i++) {
    SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
    SET_VECTOR_ELT(result, i, nanoarrow_c_infer_ptype(child_xptr));
    UNPROTECT(1);

    struct ArrowSchema* schema = schema_from_array_xptr(child_xptr);
    if (schema->name != NULL) {
      SET_STRING_ELT(result_names, i, Rf_mkCharCE(schema->name, CE_UTF8));
    } else {
      SET_STRING_ELT(result_names, i, Rf_mkChar(""));
    }
  }

  Rf_setAttrib(result, R_ClassSymbol, Rf_mkString("data.frame"));
  Rf_setAttrib(result, R_NamesSymbol, result_names);
  SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(rownames)[0] = NA_INTEGER;
  INTEGER(rownames)[1] = 0;
  Rf_setAttrib(result, R_RowNamesSymbol, rownames);
  UNPROTECT(3);
  return result;
}

SEXP nanoarrow_c_infer_ptype(SEXP array_xptr) {
  enum VectorType vector_type = nanoarrow_infer_vector_type_array(array_xptr);
  SEXP ptype = R_NilValue;

  switch (vector_type) {
    case VECTOR_TYPE_UNSPECIFIED:
    case VECTOR_TYPE_LGL:
    case VECTOR_TYPE_INT:
    case VECTOR_TYPE_DBL:
    case VECTOR_TYPE_CHR:
      ptype = PROTECT(nanoarrow_alloc_type(vector_type, 0));
      break;
    case VECTOR_TYPE_DATA_FRAME:
      ptype = PROTECT(infer_ptype_data_frame(array_xptr));
      break;
    default:
      ptype = PROTECT(call_infer_ptype_other(array_xptr));
      break;
  }

  UNPROTECT(1);
  return ptype;
}
