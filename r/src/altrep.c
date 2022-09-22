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
#include "nanoarrow.h"

#ifdef HAS_ALTREP

static R_xlen_t nanoarrow_altrep_length(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return Rf_xlength(R_altrep_data2(altrep_sexp));
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  return array_view->array->length;
}

static const void* nanoarrow_altrep_dataptr_or_null(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return DATAPTR_OR_NULL(R_altrep_data2(altrep_sexp));
  }

  return NULL;
}

static Rboolean nanoarrow_altrep_inspect(SEXP altrep_sexp, int pre, int deep, int pvec,
                                         void (*inspect_subtree)(SEXP, int, int, int)) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  const char* materialized = "";
  if (array_view_xptr == R_NilValue) {
    materialized = "materialized ";
  }

  R_xlen_t len = nanoarrow_altrep_length(altrep_sexp);
  SEXP data_class_sym = CAR(ATTRIB(ALTREP_CLASS(altrep_sexp)));
  const char* data_class_ptr = CHAR(PRINTNAME(data_class_sym));

  Rprintf("<%s%s[%ld]>\n", materialized, data_class_ptr, (long)len);
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

static void nanoarrow_altstring_set_elt(SEXP altrep_sexp, R_xlen_t i, SEXP char_sexp) {
  Rf_error("Can't SET_STRING_ELT() on nanoarrow_altstring");
}

static SEXP nanoarrow_altstring_materialize(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return R_altrep_data2(altrep_sexp);
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  SEXP result = PROTECT(Rf_allocVector(STRSXP, array_view->array->length));
  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < array_view->array->length; i++) {
    if (ArrowArrayViewIsNull(array_view, i)) {
      SET_STRING_ELT(result, i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(array_view, i);
      SET_STRING_ELT(result, i, Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
    }
  }

  R_set_altrep_data2(altrep_sexp, result);
  R_set_altrep_data1(altrep_sexp, R_NilValue);
  UNPROTECT(1);
  return result;
}

static void* nanoarrow_altrep_dataptr(SEXP altrep_sexp, Rboolean writable) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return DATAPTR(R_altrep_data2(altrep_sexp));
  }

  return DATAPTR(nanoarrow_altstring_materialize(altrep_sexp));
}

static int nanoarrow_altstring_no_na(SEXP altrep_sexp) {
  SEXP array_view_xptr = R_altrep_data1(altrep_sexp);
  if (array_view_xptr == R_NilValue) {
    return STRING_NO_NA(R_altrep_data2(altrep_sexp));
  }

  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  if (array_view->array->null_count == 0 || array_view->array->buffers[0] == NULL) {
    return TRUE;
  }

  if (array_view->array->null_count == -1) {
    const uint8_t* validity_buffer = (const uint8_t*)array_view->array->buffers[0];
    array_view->array->null_count = ArrowBitCountSet(
        validity_buffer, array_view->array->offset, array_view->array->length);
  }

  return array_view->array->null_count == 0;
}

static R_altrep_class_t nanoarrow_altrep_string_cls;

static void register_nanoarrow_altstring(DllInfo* info) {
  nanoarrow_altrep_string_cls =
      R_make_altstring_class("nanoarrow::array_string", "nanoarrow", info);
  R_set_altrep_Length_method(nanoarrow_altrep_string_cls, &nanoarrow_altrep_length);
  R_set_altrep_Inspect_method(nanoarrow_altrep_string_cls, &nanoarrow_altrep_inspect);
  R_set_altvec_Dataptr_or_null_method(nanoarrow_altrep_string_cls,
                                      &nanoarrow_altrep_dataptr_or_null);
  R_set_altvec_Dataptr_method(nanoarrow_altrep_string_cls, &nanoarrow_altrep_dataptr);

  R_set_altstring_Elt_method(nanoarrow_altrep_string_cls, &nanoarrow_altstring_elt);
  R_set_altstring_Set_elt_method(nanoarrow_altrep_string_cls,
                                 &nanoarrow_altstring_set_elt);
  R_set_altstring_No_NA_method(nanoarrow_altrep_string_cls, &nanoarrow_altstring_no_na);
}

void register_nanoarrow_altrep(DllInfo* info) { register_nanoarrow_altstring(info); }

SEXP nanoarrow_c_make_altstring(SEXP array_view_xptr) {
  struct ArrowArrayView* array_view =
      (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      break;
    default:
      return R_NilValue;
  }

  Rf_setAttrib(array_view_xptr, R_ClassSymbol, Rf_mkString("nanoarrow::array_string"));
  SEXP out =
      PROTECT(R_new_altrep(nanoarrow_altrep_string_cls, array_view_xptr, R_NilValue));
  MARK_NOT_MUTABLE(out);
  UNPROTECT(1);
  return out;
}

SEXP nanoarrow_c_is_altrep(SEXP x_sexp) {
  if (!ALTREP(x_sexp)) {
    return Rf_ScalarLogical(FALSE);
  }

  SEXP data_class_sym = CAR(ATTRIB(ALTREP_CLASS(x_sexp)));
  const char* data_class_ptr = CHAR(PRINTNAME(data_class_sym));
  return Rf_ScalarLogical(strncmp(data_class_ptr, "nanoarrow::", 11) == 0);
}

SEXP nanoarrow_c_altrep_is_materialized(SEXP x_sexp) {
  if (!ALTREP(x_sexp)) {
    return Rf_ScalarLogical(NA_LOGICAL);
  }

  SEXP data_class_sym = CAR(ATTRIB(ALTREP_CLASS(x_sexp)));
  const char* data_class_ptr = CHAR(PRINTNAME(data_class_sym));
  if (strncmp(data_class_ptr, "nanoarrow::", 11) != 0) {
    return Rf_ScalarLogical(NA_LOGICAL);
  }

  return Rf_ScalarLogical(R_altrep_data1(x_sexp) == R_NilValue);
}

SEXP nanoarrow_c_altrep_force_materialize(SEXP x_sexp, SEXP recursive_sexp) {
  if (TYPEOF(x_sexp) == VECSXP && LOGICAL(recursive_sexp)[0]) {
    int n_materialized = 0;
    for (R_xlen_t i = 0; i < Rf_xlength(x_sexp); i++) {
      SEXP n_materialized_sexp = PROTECT(
          nanoarrow_c_altrep_force_materialize(VECTOR_ELT(x_sexp, i), recursive_sexp));
      n_materialized += INTEGER(n_materialized_sexp)[0];
      UNPROTECT(1);
    }
    return Rf_ScalarInteger(n_materialized);
  }

  if (!ALTREP(x_sexp)) {
    return Rf_ScalarInteger(0);
  }

  SEXP data1 = R_altrep_data1(x_sexp);
  if (Rf_inherits(data1, "nanoarrow::array_string") && data1 != R_NilValue) {
    nanoarrow_altstring_materialize(x_sexp);
    return Rf_ScalarInteger(1);
  } else {
    return Rf_ScalarInteger(0);
  }
}

#endif
