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
#include "convert.h"
#include "util.h"

// The common case of converting a single array into a single vector is
// defined here, powered by the generic conversion available via
// convert.h but special-casing the common case of "just use the defaults"
// (i.e., no need to allocate a zero-size ptype) and returning ALTREP
// where possible.

// borrow nanoarrow_c_infer_ptype() from infer_ptype.c
SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr);
enum VectorType nanoarrow_infer_vector_type_array(SEXP array_xptr);

// This calls nanoarrow::convert_array() (via a package helper) to try S3
// dispatch to find a convert_array() method (or error if there
// isn't one)
static SEXP call_convert_array(SEXP array_xptr, SEXP ptype_sexp) {
  SEXP fun = PROTECT(Rf_install("convert_fallback_other"));
  // offset/length don't need to be modified in this case
  SEXP call = PROTECT(Rf_lang5(fun, array_xptr, R_NilValue, R_NilValue, ptype_sexp));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));
  UNPROTECT(3);
  return result;
}

// Call stop_cant_convert_array(), which gives a more informative error
// message than we can provide in a reasonable amount of C code here.
// Because we opportunistically avoid allocating a ptype object, we might
// have to allocate one here.
static void call_stop_cant_convert_array(SEXP array_xptr, enum VectorType type,
                                         SEXP ptype_sexp) {
  SEXP fun = PROTECT(Rf_install("stop_cant_convert_array"));

  if (ptype_sexp == R_NilValue) {
    ptype_sexp = PROTECT(nanoarrow_alloc_type(type, 0));
    SEXP call = PROTECT(Rf_lang3(fun, array_xptr, ptype_sexp));
    Rf_eval(call, nanoarrow_ns_pkg);
    UNPROTECT(3);
  } else {
    SEXP call = PROTECT(Rf_lang3(fun, array_xptr, ptype_sexp));
    Rf_eval(call, nanoarrow_ns_pkg);
    UNPROTECT(2);
  }
}

static SEXP convert_array_default(SEXP array_xptr, enum VectorType vector_type,
                                  SEXP ptype) {
  SEXP converter_xptr;
  if (ptype == R_NilValue) {
    converter_xptr = PROTECT(nanoarrow_converter_from_type(vector_type));
  } else {
    converter_xptr = PROTECT(nanoarrow_converter_from_ptype(ptype));
  }

  if (nanoarrow_converter_set_schema(converter_xptr, array_xptr_get_schema(array_xptr)) !=
      NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  if (nanoarrow_converter_set_array(converter_xptr, array_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  if (nanoarrow_converter_materialize_all(converter_xptr) != NANOARROW_OK) {
    call_stop_cant_convert_array(array_xptr, vector_type, ptype);
  }

  if (nanoarrow_converter_finalize(converter_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  SEXP result = PROTECT(nanoarrow_converter_release_result(converter_xptr));
  UNPROTECT(2);
  return result;
}

static SEXP convert_array_chr(SEXP array_xptr, SEXP ptype_sexp) {
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);
  struct ArrowSchemaView schema_view;
  if (ArrowSchemaViewInit(&schema_view, schema, NULL) != NANOARROW_OK) {
    Rf_error("Invalid schema");
  }

  // If array_xptr is an extension, use default conversion
  int source_can_altrep;
  switch (schema_view.type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      source_can_altrep = 1;
      break;
    default:
      source_can_altrep = 0;
  }

  if (!source_can_altrep || schema_view.extension_name.size_bytes > 0) {
    // Default conversion requires a ptype: resolve it if not already specified
    if (ptype_sexp == R_NilValue) {
      ptype_sexp = PROTECT(nanoarrow_c_infer_ptype(array_xptr_get_schema(array_xptr)));
      SEXP default_result =
          PROTECT(convert_array_default(array_xptr, VECTOR_TYPE_CHR, ptype_sexp));
      UNPROTECT(2);
      return default_result;
    } else {
      return convert_array_default(array_xptr, VECTOR_TYPE_CHR, ptype_sexp);
    }
  }

  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array->dictionary == NULL) {
    SEXP result = PROTECT(nanoarrow_c_make_altrep_chr(array_xptr));
    if (result == R_NilValue) {
      call_stop_cant_convert_array(array_xptr, VECTOR_TYPE_CHR, R_NilValue);
    }
    UNPROTECT(1);
    return result;
  } else {
    return convert_array_default(array_xptr, VECTOR_TYPE_CHR, R_NilValue);
  }
}

SEXP nanoarrow_c_convert_array(SEXP array_xptr, SEXP ptype_sexp);

static SEXP convert_array_data_frame(SEXP array_xptr, SEXP ptype_sexp) {
  struct ArrowSchema* schema = schema_from_array_xptr(array_xptr);
  struct ArrowSchemaView schema_view;
  if (ArrowSchemaViewInit(&schema_view, schema, NULL) != NANOARROW_OK) {
    Rf_error("Invalid schema");
  }

  // If array_xptr is an extension, union, or the ptype isn't a data.frame
  // use convert/materialize convert behaviour.
  // Default conversion requires a ptype: resolve it if not already specified
  if (schema_view.storage_type != NANOARROW_TYPE_STRUCT ||
      schema_view.extension_name.size_bytes > 0 ||
      (ptype_sexp != R_NilValue && !Rf_inherits(ptype_sexp, "data.frame"))) {
    if (ptype_sexp == R_NilValue) {
      ptype_sexp = PROTECT(nanoarrow_c_infer_ptype(array_xptr_get_schema(array_xptr)));
      SEXP default_result =
          PROTECT(convert_array_default(array_xptr, VECTOR_TYPE_OTHER, ptype_sexp));
      UNPROTECT(2);
      return default_result;
    } else {
      return convert_array_default(array_xptr, VECTOR_TYPE_DATA_FRAME, ptype_sexp);
    }
  }

  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  R_xlen_t n_col = array->n_children;
  SEXP result = PROTECT(Rf_allocVector(VECSXP, n_col));

  if (ptype_sexp == R_NilValue) {
    SEXP result_names = PROTECT(Rf_allocVector(STRSXP, n_col));

    for (R_xlen_t i = 0; i < n_col; i++) {
      SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
      SET_VECTOR_ELT(result, i, nanoarrow_c_convert_array(child_xptr, R_NilValue));
      UNPROTECT(1);

      struct ArrowSchema* schema = schema_from_array_xptr(child_xptr);
      if (schema->name != NULL) {
        SET_STRING_ELT(result_names, i, Rf_mkCharCE(schema->name, CE_UTF8));
      } else {
        SET_STRING_ELT(result_names, i, Rf_mkChar(""));
      }
    }

    Rf_setAttrib(result, R_NamesSymbol, result_names);
    Rf_setAttrib(result, R_ClassSymbol, nanoarrow_cls_data_frame);
    UNPROTECT(1);
  } else {
    if (n_col != Rf_xlength(ptype_sexp)) {
      Rf_error("Expected data.frame() ptype with %ld column(s) but found %ld column(s)",
               (long)n_col, (long)Rf_xlength(ptype_sexp));
    }

    for (R_xlen_t i = 0; i < n_col; i++) {
      SEXP child_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
      SEXP child_ptype = VECTOR_ELT(ptype_sexp, i);
      SET_VECTOR_ELT(result, i, nanoarrow_c_convert_array(child_xptr, child_ptype));
      UNPROTECT(1);
    }

    Rf_setAttrib(result, R_NamesSymbol, Rf_getAttrib(ptype_sexp, R_NamesSymbol));
    Rf_copyMostAttrib(ptype_sexp, result);
  }

  if (Rf_inherits(result, "data.frame")) {
    nanoarrow_set_rownames(result, array->length);
  }

  UNPROTECT(1);
  return result;
}

SEXP nanoarrow_c_convert_array(SEXP array_xptr, SEXP ptype_sexp) {
  // See if we can skip any ptype resolution at all
  if (ptype_sexp == R_NilValue) {
    enum VectorType vector_type = nanoarrow_infer_vector_type_array(array_xptr);
    switch (vector_type) {
      case VECTOR_TYPE_LGL:
      case VECTOR_TYPE_INT:
      case VECTOR_TYPE_DBL:
        return convert_array_default(array_xptr, vector_type, R_NilValue);
      case VECTOR_TYPE_CHR:
        return convert_array_chr(array_xptr, ptype_sexp);
      case VECTOR_TYPE_DATA_FRAME:
        return convert_array_data_frame(array_xptr, R_NilValue);
      default:
        break;
    }

    // Otherwise, resolve the ptype and use it (this will also error
    // for ptypes that can't be resolved)
    ptype_sexp = PROTECT(nanoarrow_c_infer_ptype(array_xptr_get_schema(array_xptr)));
    SEXP result = nanoarrow_c_convert_array(array_xptr, ptype_sexp);
    UNPROTECT(1);
    return result;
  }

  // Handle some S3 objects internally to avoid S3 dispatch
  // (e.g., when looping over a data frame with a lot of columns)
  if (Rf_isObject(ptype_sexp)) {
    if (nanoarrow_ptype_is_data_frame(ptype_sexp)) {
      return convert_array_data_frame(array_xptr, ptype_sexp);
    } else if (Rf_inherits(ptype_sexp, "vctrs_unspecified") ||
               Rf_inherits(ptype_sexp, "blob") ||
               Rf_inherits(ptype_sexp, "vctrs_list_of") ||
               Rf_inherits(ptype_sexp, "Date") || Rf_inherits(ptype_sexp, "hms") ||
               Rf_inherits(ptype_sexp, "POSIXct") ||
               Rf_inherits(ptype_sexp, "difftime") ||
               Rf_inherits(ptype_sexp, "integer64")) {
      return convert_array_default(array_xptr, VECTOR_TYPE_UNINITIALIZED, ptype_sexp);
    } else {
      return call_convert_array(array_xptr, ptype_sexp);
    }
  }

  // If we're here, these are non-S3 objects
  switch (TYPEOF(ptype_sexp)) {
    case LGLSXP:
      return convert_array_default(array_xptr, VECTOR_TYPE_LGL, ptype_sexp);
    case INTSXP:
      return convert_array_default(array_xptr, VECTOR_TYPE_INT, ptype_sexp);
    case REALSXP:
      return convert_array_default(array_xptr, VECTOR_TYPE_DBL, ptype_sexp);
    case STRSXP:
      return convert_array_chr(array_xptr, ptype_sexp);
    default:
      return call_convert_array(array_xptr, ptype_sexp);
  }
}
