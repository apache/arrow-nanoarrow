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

enum VectorType {
  VECTOR_TYPE_LGL,
  VECTOR_TYPE_INT,
  VECTOR_TYPE_DBL,
  VECTOR_TYPE_CHR,
  VECTOR_TYPE_DATA_FRAME,
  VECTOR_TYPE_LIST_OF_RAW,
  VECTOR_TYPE_UNKNOWN
};

// These conversions are the conversions we can be sure about without inspecting
// any extra data from schema or the array.
static enum VectorType vector_type_from_array_type(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_BOOL:
      return VECTOR_TYPE_LGL;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
      return VECTOR_TYPE_INT;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      return VECTOR_TYPE_DBL;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      return VECTOR_TYPE_CHR;

    case NANOARROW_TYPE_STRUCT:
      return VECTOR_TYPE_DATA_FRAME;

    default:
      return VECTOR_TYPE_UNKNOWN;
  }
}

static enum VectorType vector_type_from_array_xptr(SEXP array_xptr) {
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);

  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  if (ArrowSchemaViewInit(&schema_view, schema, &error) != NANOARROW_OK) {
    Rf_error("vector_type_from_array_view_xptr(): %s", ArrowErrorMessage(&error));
  }

  // Try the types with a definitive conversion
  enum VectorType result = vector_type_from_array_type(schema_view.data_type);
  if (result != VECTOR_TYPE_UNKNOWN) {
    return result;
  }

  // TODO: Try inspecting the schema/array (e.g., to range-check an int64
  // to check if it's coercible to int or double)
  return VECTOR_TYPE_UNKNOWN;
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
  enum VectorType vector_type = vector_type_from_array_xptr(array_xptr);

  switch (vector_type) {
    case VECTOR_TYPE_LGL:
      return Rf_allocVector(LGLSXP, 0);
    case VECTOR_TYPE_INT:
      return Rf_allocVector(INTSXP, 0);
    case VECTOR_TYPE_DBL:
      return Rf_allocVector(REALSXP, 0);
    case VECTOR_TYPE_CHR:
      return Rf_allocVector(STRSXP, 0);
    case VECTOR_TYPE_DATA_FRAME:
      return infer_ptype_data_frame(array_xptr);
    default:
      Rf_error("Can't guess default ptype for array");
  }

  return R_NilValue;
}

static SEXP call_from_nanoarrow_array(SEXP array_xptr, SEXP ptype_sexp) {
  // Calls nanoarrow::from_nanoarrow_array() with .call_from_c = TRUE or
  // something to make sure we don't go in circles
  Rf_error("Can't convert array to vector");
}

SEXP nanoarrow_c_from_array(SEXP array_xptr, SEXP ptype_sexp);

static SEXP from_array_to_data_frame(SEXP array_xptr, SEXP ptype_sexp) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  R_xlen_t n_col = array->n_children;
  SEXP result = PROTECT(Rf_allocVector(VECSXP, n_col));

  if (ptype_sexp == R_NilValue) {
    SEXP result_names = PROTECT(Rf_allocVector(STRSXP, n_col));

    for (R_xlen_t i = 0; i < n_col; i++) {
      SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
      SET_VECTOR_ELT(result, i, nanoarrow_c_from_array(child_xptr, R_NilValue));
      UNPROTECT(1);

      struct ArrowSchema* schema = schema_from_array_xptr(child_xptr);
      if (schema->name != NULL) {
        SET_STRING_ELT(result_names, i, Rf_mkCharCE(schema->name, CE_UTF8));
      } else {
        SET_STRING_ELT(result_names, i, Rf_mkChar(""));
      }
    }

    Rf_setAttrib(result, R_NamesSymbol, result_names);
    UNPROTECT(1);
  } else {
    if (n_col != Rf_xlength(ptype_sexp)) {
      Rf_error("Expected data.frame() ptype with %ld column(s) but found %ld column(s)",
               (long)n_col, (long)Rf_xlength(ptype_sexp));
    }

    for (R_xlen_t i = 0; i < n_col; i++) {
      SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
      SEXP child_ptype = VECTOR_ELT(ptype_sexp, i);
      SET_VECTOR_ELT(result, i, nanoarrow_c_from_array(child_xptr, child_ptype));
      UNPROTECT(1);
    }

    Rf_setAttrib(result, R_NamesSymbol, Rf_getAttrib(ptype_sexp, R_NamesSymbol));
  }

  Rf_setAttrib(result, R_ClassSymbol, Rf_mkString("data.frame"));
  SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(rownames)[0] = NA_INTEGER;
  INTEGER(rownames)[1] = array->length;
  Rf_setAttrib(result, R_RowNamesSymbol, rownames);

  UNPROTECT(2);
  return result;
}

static SEXP from_array_to_int(SEXP array_xptr) {
  SEXP array_view_xptr = PROTECT(array_view_xptr_from_array_xptr(array_xptr));
  SEXP result = PROTECT(nanoarrow_materialize_int(array_view_from_xptr(array_view_xptr)));
  UNPROTECT(2);
  return result;
}

static SEXP from_array_to_chr(SEXP array_xptr) {
  SEXP array_view_xptr = PROTECT(array_view_xptr_from_array_xptr(array_xptr));
  SEXP result = PROTECT(nanoarrow_c_make_altrep_chr(array_view_xptr));
  if (result == R_NilValue) {
    Rf_error("Can't convert array to character()");
  }
  UNPROTECT(2);
  return result;
}

SEXP nanoarrow_c_from_array(SEXP array_xptr, SEXP ptype_sexp) {
  // See if we can skip any ptype resolution at all
  if (ptype_sexp == R_NilValue) {
    enum VectorType vector_type = vector_type_from_array_xptr(array_xptr);
    switch (vector_type) {
      case VECTOR_TYPE_INT:
        return from_array_to_int(array_xptr);
      case VECTOR_TYPE_CHR:
        return from_array_to_chr(array_xptr);
      case VECTOR_TYPE_DATA_FRAME:
        return from_array_to_data_frame(array_xptr, R_NilValue);
      default:
        break;
    }
  }

  // Handle some S3 objects internally to avoid S3 dispatch
  // (e.g., when looping over a data frame with a lot of columns)
  if (Rf_isObject(ptype_sexp)) {
    if (Rf_inherits(ptype_sexp, "data.frame")) {
      return from_array_to_data_frame(array_xptr, ptype_sexp);
    } else {
      return call_from_nanoarrow_array(array_xptr, ptype_sexp);
    }
  }

  // If we're here, these are non-S3 objects
  switch (TYPEOF(ptype_sexp)) {
    case INTSXP:
      return from_array_to_int(array_xptr);
    case STRSXP:
      return from_array_to_chr(array_xptr);
    default:
      return call_from_nanoarrow_array(array_xptr, ptype_sexp);
  }
}
