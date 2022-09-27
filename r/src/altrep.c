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

#include <stdlib.h>
#include <string.h>

#include "altrep.h"
#include "array.h"
#include "nanoarrow.h"

#include "materialize.h"

#ifdef HAS_ALTREP

// This file defines all ALTREP classes used to speed up conversion
// from an arrow_array to an R vector. Currently only string and
// large string arrays are converted to ALTREP.
//
// All ALTREP classes follow some common patterns:
//
// - R_altrep_data1() holds an external pointer to a struct ArrowArrayView
// - R_altrep_data2() holds the materialized version of the vector.
// - When materialization happens, we set R_altrep_data1() to R_NilValue
//   to ensure we don't hold on to any more resources than needed.

static R_xlen_t nanoarrow_altrep_length(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return Rf_xlength(R_altrep_data2(altrep_sexp));
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  return array_view->array->length;
}

static Rboolean nanoarrow_altrep_inspect(SEXP altrep_sexp, int pre, int deep, int pvec,
                                         void (*inspect_subtree)(SEXP, int, int, int)) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  const char* materialized = "";
  if (array_view_xptr == R_NilValue) {
    materialized = "materialized ";
  }

  R_xlen_t len = nanoarrow_altrep_length(altrep_sexp);
  const char* class_name = nanoarrow_altrep_class(altrep_sexp);
  Rprintf("<%s%s[%ld]>\n", materialized, class_name, (long)len);
  return TRUE;
}

static SEXP nanoarrow_altstring_elt(SEXP altrep_sexp, R_xlen_t i) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return STRING_ELT(R_altrep_data2(altrep_sexp), i);
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  if (ArrowArrayViewIsNull(array_view, i)) {
    return NA_STRING;
  }

  struct ArrowStringView item = ArrowArrayViewGetStringUnsafe(array_view, i);
  return Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8);
}

static SEXP nanoarrow_altstring_materialize(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return R_altrep_data2(altrep_sexp);
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  SEXP result = PROTECT(nanoarrow_materialize_chr(array_view));
  R_set_altrep_data2(altrep_sexp, result);
  R_set_altrep_data1(altrep_sexp, R_NilValue);
  UNPROTECT(1);
  return result;
}

static void* nanoarrow_altrep_dataptr(SEXP altrep_sexp, Rboolean writable) {
  return DATAPTR(nanoarrow_altstring_materialize(altrep_sexp));
}

static const void* nanoarrow_altrep_dataptr_or_null(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return DATAPTR_OR_NULL(R_altrep_data2(altrep_sexp));
  }

  return NULL;
}

static R_altrep_class_t nanoarrow_altrep_chr_cls;

#endif

static void register_nanoarrow_altstring(DllInfo* info) {
#ifdef HAS_ALTREP
  nanoarrow_altrep_chr_cls =
      R_make_altstring_class("nanoarrow::altrep_chr", "nanoarrow", info);
  R_set_altrep_Length_method(nanoarrow_altrep_chr_cls, &nanoarrow_altrep_length);
  R_set_altrep_Inspect_method(nanoarrow_altrep_chr_cls, &nanoarrow_altrep_inspect);
  R_set_altvec_Dataptr_or_null_method(nanoarrow_altrep_chr_cls,
                                      &nanoarrow_altrep_dataptr_or_null);
  R_set_altvec_Dataptr_method(nanoarrow_altrep_chr_cls, &nanoarrow_altrep_dataptr);

  R_set_altstring_Elt_method(nanoarrow_altrep_chr_cls, &nanoarrow_altstring_elt);

  // Notes about other available methods:
  //
  // - The no_na method never seems to get called (anyNA() doesn't seem to
  //   use it)
  // - Because set_Elt is not defined, SET_STRING_ELT() will modify the
  //   technically modify the materialized value. The object has been marked
  //   immutable but in the case of a string this is fine because we materialize
  //   when this happens (via Dataptr).
  // - It may be beneficial to implement the Extract_subset method to defer string
  //   conversion even longer since this is expensive compared to rearranging integer
  //   indices.
#endif
}

void register_nanoarrow_altrep(DllInfo* info) { register_nanoarrow_altstring(info); }

SEXP nanoarrow_c_make_altrep_chr(SEXP array_view_xptr) {
#ifdef HAS_ALTREP
  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      break;
    default:
      return R_NilValue;
  }

  // Ensure the array that we're attaching to this ALTREP object does not keep its
  // parent struct alive unnecessarily (i.e., a user can select only a few columns
  // and the memory for the unused columns will be released).
  SEXP array_xptr_independent =
      PROTECT(array_xptr_ensure_independent(R_ExternalPtrProtected(array_view_xptr)));
  array_view->array = array_from_xptr(array_xptr_independent);
  R_SetExternalPtrProtected(array_view_xptr, array_xptr_independent);
  UNPROTECT(1);

  Rf_setAttrib(array_view_xptr, R_ClassSymbol, Rf_mkString("nanoarrow::altrep_chr"));
  SEXP out = PROTECT(R_new_altrep(nanoarrow_altrep_chr_cls, array_view_xptr, R_NilValue));
  MARK_NOT_MUTABLE(out);
  UNPROTECT(1);
  return out;
#else
  return R_NilValue;
#endif
}

SEXP nanoarrow_c_is_altrep(SEXP x_sexp) {
  return Rf_ScalarLogical(is_nanoarrow_altrep(x_sexp));
}

SEXP nanoarrow_c_altrep_is_materialized(SEXP x_sexp) {
  const char* class_name = nanoarrow_altrep_class(x_sexp);
  if (class_name == NULL || strncmp(class_name, "nanoarrow::", 11) != 0) {
    return Rf_ScalarLogical(NA_LOGICAL);
  } else {
    return Rf_ScalarLogical(R_altrep_data1(x_sexp) == R_NilValue);
  }
}

SEXP nanoarrow_c_altrep_force_materialize(SEXP x_sexp, SEXP recursive_sexp) {
  // The recursive flag lets a developer/user force materialization of any
  // string columns in a data.frame that came from nanoarrow.
  if (Rf_inherits(x_sexp, "data.frame") && LOGICAL(recursive_sexp)[0]) {
    int n_materialized = 0;
    for (R_xlen_t i = 0; i < Rf_xlength(x_sexp); i++) {
      SEXP n_materialized_sexp = PROTECT(
          nanoarrow_c_altrep_force_materialize(VECTOR_ELT(x_sexp, i), recursive_sexp));
      n_materialized += INTEGER(n_materialized_sexp)[0];
      UNPROTECT(1);
    }
    return Rf_ScalarInteger(n_materialized);
  }

  const char* class_name = nanoarrow_altrep_class(x_sexp);
  if (class_name && strcmp(class_name, "nanoarrow::altrep_chr") == 0) {
    // Force materialization even if already materialized (the method
    // should be safe to call more than once as written here)
    int already_materialized = R_altrep_data1(x_sexp) == R_NilValue;
    nanoarrow_altstring_materialize(x_sexp);
    return Rf_ScalarInteger(!already_materialized);
  } else {
    return Rf_ScalarInteger(0);
  }
}
