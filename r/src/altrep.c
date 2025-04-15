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
#include "convert.h"
#include "nanoarrow.h"
#include "util.h"

// This file defines all ALTREP classes used to speed up conversion
// from an arrow_array to an R vector. Currently only string and
// large string arrays are converted to ALTREP.
//
// All ALTREP classes follow some common patterns:
//
// - R_altrep_data1() holds an external pointer to a struct RConverter.
// - R_altrep_data2() holds the materialized version of the vector.
// - When materialization happens, we set R_altrep_data1() to R_NilValue
//   to ensure we don't hold on to any more resources than needed.

static R_xlen_t nanoarrow_altrep_length(SEXP altrep_sexp) {
  SEXP converter_xptr = R_altrep_data1(altrep_sexp);
  if (converter_xptr == R_NilValue) {
    return Rf_xlength(R_altrep_data2(altrep_sexp));
  }

  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  return converter->array_view.array->length;
}

static Rboolean nanoarrow_altrep_inspect(SEXP altrep_sexp, int pre, int deep, int pvec,
                                         void (*inspect_subtree)(SEXP, int, int, int)) {
  SEXP converter_xptr = R_altrep_data1(altrep_sexp);
  const char* materialized = "";
  if (converter_xptr == R_NilValue) {
    materialized = "materialized ";
  }

  R_xlen_t len = nanoarrow_altrep_length(altrep_sexp);
  const char* class_name = nanoarrow_altrep_class(altrep_sexp);
  Rprintf("<%s%s[%ld]>\n", materialized, class_name, (long)len);
  return TRUE;
}

static SEXP nanoarrow_altstring_elt(SEXP altrep_sexp, R_xlen_t i) {
  SEXP converter_xptr = R_altrep_data1(altrep_sexp);
  if (converter_xptr == R_NilValue) {
    return STRING_ELT(R_altrep_data2(altrep_sexp), i);
  }

  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  if (ArrowArrayViewIsNull(&converter->array_view, i)) {
    return NA_STRING;
  }

  struct ArrowStringView item = ArrowArrayViewGetStringUnsafe(&converter->array_view, i);
  return Rf_mkCharLenCE(item.data, (int)item.size_bytes, CE_UTF8);
}

static SEXP nanoarrow_altstring_materialize(SEXP altrep_sexp) {
  SEXP converter_xptr = R_altrep_data1(altrep_sexp);
  if (converter_xptr == R_NilValue) {
    return R_altrep_data2(altrep_sexp);
  }

  if (nanoarrow_converter_materialize_all(converter_xptr) != NANOARROW_OK) {
    Rf_error("Error materializing altstring");
  }

  if (nanoarrow_converter_finalize(converter_xptr) != NANOARROW_OK) {
    Rf_error("Error finalizing materialized altstring");
  }

  SEXP result_sexp = PROTECT(nanoarrow_converter_release_result(converter_xptr));
  R_set_altrep_data2(altrep_sexp, result_sexp);
  R_set_altrep_data1(altrep_sexp, R_NilValue);
  UNPROTECT(1);
  return result_sexp;
}

static void* nanoarrow_altrep_dataptr(SEXP altrep_sexp, Rboolean writable) {
  // DATAPTR() can't be called in R >= 4.5.0 without a check NOTE, but
  // there doesn't appear to be an alternative to support an ALTREP string
  // class that can materialize.
  return (void*)DATAPTR_RO(nanoarrow_altstring_materialize(altrep_sexp));
}

static const void* nanoarrow_altrep_dataptr_or_null(SEXP altrep_sexp) {
  SEXP converter_xptr = R_altrep_data1(altrep_sexp);
  if (converter_xptr == R_NilValue) {
    return DATAPTR_OR_NULL(R_altrep_data2(altrep_sexp));
  }

  return NULL;
}

static R_altrep_class_t nanoarrow_altrep_chr_cls;

static void register_nanoarrow_altstring(DllInfo* info) {
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
  // - The duplicate method may be useful because it's used when setting attributes
  //   or unclassing the vector.
}

void register_nanoarrow_altrep(DllInfo* info) { register_nanoarrow_altstring(info); }

SEXP nanoarrow_c_make_altrep_chr(SEXP array_xptr) {
  SEXP schema_xptr = array_xptr_get_schema(array_xptr);

  // Create the converter
  SEXP converter_xptr = PROTECT(nanoarrow_converter_from_type(VECTOR_TYPE_CHR));
  if (nanoarrow_converter_set_schema(converter_xptr, schema_xptr) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  switch (converter->array_view.storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      break;
    default:
      UNPROTECT(1);
      return R_NilValue;
  }

  // Ensure the array that we're attaching to this ALTREP object does not keep its
  // parent struct alive unnecessarily (i.e., a user can select only a few columns
  // and the memory for the unused columns will be released).
  SEXP array_xptr_independent = PROTECT(array_xptr_ensure_independent(array_xptr));

  if (nanoarrow_converter_set_array(converter_xptr, array_xptr_independent) !=
      NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  Rf_setAttrib(converter_xptr, R_ClassSymbol, nanoarrow_cls_altrep_chr);
  SEXP out = PROTECT(R_new_altrep(nanoarrow_altrep_chr_cls, converter_xptr, R_NilValue));
  MARK_NOT_MUTABLE(out);
  UNPROTECT(3);
  return out;
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
