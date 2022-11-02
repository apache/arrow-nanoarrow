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

// This calls nanoarrow::materialize_array() (via a package helper) to try S3
// dispatch to find a materialize_array() method (or error if there
// isn't one)
static SEXP call_materialize_array(SEXP array_xptr, SEXP ptype_sexp) {
  SEXP ns = PROTECT(R_FindNamespace(Rf_mkString("nanoarrow")));
  SEXP call =
      PROTECT(Rf_lang3(Rf_install("materialize_array_from_c"), array_xptr, ptype_sexp));
  SEXP result = PROTECT(Rf_eval(call, ns));
  UNPROTECT(3);
  return result;
}

// Call stop_cant_materialize_array(), which gives a more informative error
// message than we can provide in a reasonable amount of C code here
static void call_stop_cant_materialize_array(SEXP array_xptr, enum VectorType type) {
  SEXP ns = PROTECT(R_FindNamespace(Rf_mkString("nanoarrow")));
  SEXP ptype_sexp = PROTECT(nanoarrow_alloc_type(type, 0));
  SEXP call = PROTECT(
      Rf_lang3(Rf_install("stop_cant_materialize_array"), array_xptr, ptype_sexp));
  Rf_eval(call, ns);
  UNPROTECT(3);
}

SEXP nanoarrow_c_materialize_array(SEXP array_xptr, SEXP ptype_sexp);

static SEXP materialize_array_data_frame(SEXP array_xptr, SEXP ptype_sexp) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  R_xlen_t n_col = array->n_children;
  SEXP result = PROTECT(Rf_allocVector(VECSXP, n_col));

  if (ptype_sexp == R_NilValue) {
    SEXP result_names = PROTECT(Rf_allocVector(STRSXP, n_col));

    for (R_xlen_t i = 0; i < n_col; i++) {
      SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
      SET_VECTOR_ELT(result, i, nanoarrow_c_materialize_array(child_xptr, R_NilValue));
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
      SET_VECTOR_ELT(result, i, nanoarrow_c_materialize_array(child_xptr, child_ptype));
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

static SEXP materialize_array_type(SEXP array_xptr, enum VectorType vector_type) {
  SEXP array_view_xptr = PROTECT(array_view_xptr_from_array_xptr(array_xptr));
  struct ArrowArrayView* array_view = array_view_from_xptr(array_view_xptr);

  SEXP result_sexp =
      PROTECT(nanoarrow_alloc_type(vector_type, array_view->array->length));

  struct ArrayViewSlice src = DefaultArrayViewSlice(array_view);
  struct VectorSlice dst = DefaultVectorSlice(result_sexp);
  struct MaterializeOptions options = DefaultMaterializeOptions();
  struct MaterializeContext context = DefaultMaterializeContext();

  if (nanoarrow_materialize(&src, &dst, &options, &context) != NANOARROW_OK) {
    call_stop_cant_materialize_array(array_xptr, vector_type);
  }

  UNPROTECT(2);
  return result_sexp;
}

static SEXP materialize_array_list_of_raw(SEXP array_xptr) {
  SEXP array_view_xptr = PROTECT(array_view_xptr_from_array_xptr(array_xptr));
  struct ArrowArrayView* array_view = array_view_from_xptr(array_view_xptr);

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      break;
    default:
      UNPROTECT(1);
      return R_NilValue;
  }

  SEXP result_sexp = PROTECT(Rf_allocVector(VECSXP, array_view->array->length));

  if (array_view->storage_type == NANOARROW_TYPE_NA) {
    UNPROTECT(2);
    return result_sexp;
  }

  struct ArrowBufferView item;
  SEXP item_sexp;
  for (R_xlen_t i = 0; i < array_view->array->length; i++) {
    if (!ArrowArrayViewIsNull(array_view, i)) {
      item = ArrowArrayViewGetBytesUnsafe(array_view, i);
      item_sexp = PROTECT(Rf_allocVector(RAWSXP, item.n_bytes));
      memcpy(RAW(item_sexp), item.data.data, item.n_bytes);
      SET_VECTOR_ELT(result_sexp, i, item_sexp);
      UNPROTECT(1);
    }
  }

  UNPROTECT(2);
  return result_sexp;
}

static SEXP materialize_array_chr(SEXP array_xptr) {
  SEXP array_view_xptr = PROTECT(array_view_xptr_from_array_xptr(array_xptr));
  SEXP result = PROTECT(nanoarrow_c_make_altrep_chr(array_view_xptr));
  if (result == R_NilValue) {
    call_stop_cant_materialize_array(array_xptr, VECTOR_TYPE_CHR);
  }
  UNPROTECT(2);
  return result;
}

// TODO: Lists are not all that well supported yet.
static SEXP materialize_array_list(SEXP array_xptr, SEXP ptype_sexp) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);

  struct ArrowSchemaView schema_view;
  struct ArrowError error;
  if (ArrowSchemaViewInit(&schema_view, schema, &error) != NANOARROW_OK) {
    Rf_error("materialize_array_list(): %s", ArrowErrorMessage(&error));
  }

  SEXP result = R_NilValue;
  switch (schema_view.data_type) {
    case NANOARROW_TYPE_NA:
      result = PROTECT(Rf_allocVector(VECSXP, array->length));
      break;
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      result = PROTECT(materialize_array_list_of_raw(array_xptr));
      break;
    default:
      call_stop_cant_materialize_array(array_xptr, VECTOR_TYPE_CHR);
  }

  UNPROTECT(1);
  return result;
}

// borrow nanoarrow_c_infer_ptype() from infer_ptype.c
SEXP nanoarrow_c_infer_ptype(SEXP array_xptr);
enum VectorType nanoarrow_infer_vector_type_array(SEXP array_xptr);

SEXP nanoarrow_c_materialize_array(SEXP array_xptr, SEXP ptype_sexp) {
  // See if we can skip any ptype resolution at all
  if (ptype_sexp == R_NilValue) {
    enum VectorType vector_type = nanoarrow_infer_vector_type_array(array_xptr);
    switch (vector_type) {
      case VECTOR_TYPE_UNSPECIFIED:
      case VECTOR_TYPE_LGL:
      case VECTOR_TYPE_INT:
      case VECTOR_TYPE_DBL:
        return materialize_array_type(array_xptr, vector_type);
      case VECTOR_TYPE_CHR:
        return materialize_array_chr(array_xptr);
      case VECTOR_TYPE_LIST_OF_RAW:
        return materialize_array_list_of_raw(array_xptr);
      case VECTOR_TYPE_DATA_FRAME:
        return materialize_array_data_frame(array_xptr, R_NilValue);
      default:
        break;
    }

    // Otherwise, resolve the ptype and use it (this will also error
    // for ptypes that can't be resolved)
    ptype_sexp = PROTECT(nanoarrow_c_infer_ptype(array_xptr));
    SEXP result = nanoarrow_c_materialize_array(array_xptr, ptype_sexp);
    UNPROTECT(1);
    return result;
  }

  // Handle some S3 objects internally to avoid S3 dispatch
  // (e.g., when looping over a data frame with a lot of columns)
  if (Rf_isObject(ptype_sexp)) {
    if (Rf_inherits(ptype_sexp, "data.frame") && !Rf_inherits(ptype_sexp, "tbl_df")) {
      return materialize_array_data_frame(array_xptr, ptype_sexp);
    } else if (Rf_inherits(ptype_sexp, "vctrs_unspecified")) {
      return materialize_array_type(array_xptr, VECTOR_TYPE_UNSPECIFIED);
    } else {
      return call_materialize_array(array_xptr, ptype_sexp);
    }
  }

  // If we're here, these are non-S3 objects
  switch (TYPEOF(ptype_sexp)) {
    case LGLSXP:
      return materialize_array_type(array_xptr, VECTOR_TYPE_LGL);
    case INTSXP:
      return materialize_array_type(array_xptr, VECTOR_TYPE_INT);
    case REALSXP:
      return materialize_array_type(array_xptr, VECTOR_TYPE_DBL);
    case STRSXP:
      return materialize_array_chr(array_xptr);
    case VECSXP:
      return materialize_array_list(array_xptr, ptype_sexp);
    default:
      return call_materialize_array(array_xptr, ptype_sexp);
  }
}
