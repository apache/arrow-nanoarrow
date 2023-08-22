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

#include "util.h"

// Needed for the list_of materializer
#include "convert.h"

#include "materialize.h"
#include "materialize_blob.h"
#include "materialize_chr.h"
#include "materialize_date.h"
#include "materialize_dbl.h"
#include "materialize_difftime.h"
#include "materialize_int.h"
#include "materialize_lgl.h"
#include "materialize_posixct.h"
#include "materialize_unspecified.h"

SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len) {
  switch (vector_type) {
    case VECTOR_TYPE_LGL:
      return Rf_allocVector(LGLSXP, len);
    case VECTOR_TYPE_INT:
      return Rf_allocVector(INTSXP, len);
    case VECTOR_TYPE_DBL:
      return Rf_allocVector(REALSXP, len);
    case VECTOR_TYPE_CHR:
      return Rf_allocVector(STRSXP, len);
    default:
      return R_NilValue;
  }
}

// A version of Rf_getAttrib(x, sym) != R_NilValue that never
// expands the row.names attribute
static int has_attrib_safe(SEXP x, SEXP sym) {
  for (SEXP atts = ATTRIB(x); atts != R_NilValue; atts = CDR(atts)) {
    if (TAG(atts) == sym) return TRUE;
  }
  return FALSE;
}

R_xlen_t nanoarrow_data_frame_size(SEXP x) {
  if (Rf_length(x) > 0) {
    // This both avoids materializing the row.names attribute and
    // makes this work with struct-style vctrs that don't have a
    // row.names attribute but that always have one or more element
    return Rf_xlength(VECTOR_ELT(x, 0));
  } else {
    // Since ALTREP was introduced, materializing the row.names attribute is
    // usually deferred such that values in the form c(NA, -nrow), 1:nrow, or
    // as.character(1:nrow) are never actually computed when the length is
    // taken.
    return Rf_xlength(Rf_getAttrib(x, R_RowNamesSymbol));
  }
}

void nanoarrow_set_rownames(SEXP x, R_xlen_t len) {
  // If len fits in the integer range, we can use the c(NA, -nrow)
  // shortcut for the row.names attribute. R expands this when
  // the actual value is accessed (even from Rf_getAttrib()).
  // If len does not fit in the integer range, we need
  // as.character(seq_len(nrow)) (which returns a deferred ALTREP
  // string conversion of an ALTREP sequence in recent R). Manipulating
  // data frames with more than INT_MAX rows is not supported in most
  // places but column access still works.
  if (len <= INT_MAX) {
    SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(rownames)[0] = NA_INTEGER;
    INTEGER(rownames)[1] = -len;
    Rf_setAttrib(x, R_RowNamesSymbol, rownames);
    UNPROTECT(1);
  } else {
    SEXP length_dbl = PROTECT(Rf_ScalarReal(len));
    SEXP seq_len_symbol = PROTECT(Rf_install("seq_len"));
    SEXP seq_len_call = PROTECT(Rf_lang2(seq_len_symbol, length_dbl));
    SEXP rownames_call = PROTECT(Rf_lang2(R_AsCharacterSymbol, seq_len_call));
    Rf_setAttrib(x, R_RowNamesSymbol, Rf_eval(rownames_call, R_BaseNamespace));
    UNPROTECT(4);
  }
}

int nanoarrow_ptype_is_data_frame(SEXP ptype) {
  return Rf_isObject(ptype) && TYPEOF(ptype) == VECSXP &&
         (Rf_inherits(ptype, "data.frame") ||
          (Rf_xlength(ptype) > 0 && has_attrib_safe(ptype, R_NamesSymbol)));
}

SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len) {
  SEXP result;

  if (Rf_isObject(ptype)) {
    if (nanoarrow_ptype_is_data_frame(ptype)) {
      R_xlen_t num_cols = Rf_xlength(ptype);
      result = PROTECT(Rf_allocVector(VECSXP, num_cols));
      for (R_xlen_t i = 0; i < num_cols; i++) {
        SET_VECTOR_ELT(result, i,
                       nanoarrow_materialize_realloc(VECTOR_ELT(ptype, i), len));
      }

      // Set attributes from ptype
      Rf_setAttrib(result, R_NamesSymbol, Rf_getAttrib(ptype, R_NamesSymbol));
      Rf_copyMostAttrib(ptype, result);

      // ...except rownames
      if (Rf_inherits(ptype, "data.frame")) {
        nanoarrow_set_rownames(result, len);
      }
    } else {
      result = PROTECT(Rf_allocVector(TYPEOF(ptype), len));
      Rf_copyMostAttrib(ptype, result);
    }
  } else {
    result = PROTECT(Rf_allocVector(TYPEOF(ptype), len));
  }

  UNPROTECT(1);
  return result;
}

// Used in union building to pre-set all values to null
static void fill_vec_with_nulls(SEXP x, R_xlen_t offset, R_xlen_t len) {
  if (nanoarrow_ptype_is_data_frame(x)) {
    for (R_xlen_t i = 0; i < Rf_xlength(x); i++) {
      fill_vec_with_nulls(VECTOR_ELT(x, i), offset, len);
    }

    return;
  }

  switch (TYPEOF(x)) {
    case LGLSXP:
    case INTSXP: {
      int* values = INTEGER(x);
      for (R_xlen_t i = 0; i < len; i++) {
        values[offset + i] = NA_INTEGER;
      }
      return;
    }
    case REALSXP: {
      double* values = REAL(x);
      for (R_xlen_t i = 0; i < len; i++) {
        values[offset + i] = NA_REAL;
      }
      return;
    }
    case STRSXP:
      for (R_xlen_t i = 0; i < len; i++) {
        SET_STRING_ELT(x, offset + i, NA_STRING);
      }
      return;
    case VECSXP:
      for (R_xlen_t i = 0; i < len; i++) {
        SET_VECTOR_ELT(x, offset + i, R_NilValue);
      }
      return;
    default:
      Rf_error("Attempt to fill vector with nulls with unsupported type");
  }
}

static int nanoarrow_materialize_other(struct RConverter* converter,
                                       SEXP converter_xptr) {
  // Ensure that we have a ptype SEXP to send in the call back to R
  if (converter->ptype_view.ptype == R_NilValue) {
    SEXP ptype = PROTECT(nanoarrow_alloc_type(converter->ptype_view.vector_type, 0));
    converter->ptype_view.ptype = ptype;
    SET_VECTOR_ELT(R_ExternalPtrProtected(converter_xptr), 0, ptype);
    UNPROTECT(1);
  }

  // A unique situation where we don't want owning external pointers because we know
  // these are protected for the duration of our call into R and because we don't want
  // the underlying array to be released and invalidate the converter. The R code in
  // convert_fallback_other() takes care of ensuring an independent copy with the correct
  // offset/length.
  SEXP schema_xptr =
      PROTECT(R_MakeExternalPtr(converter->schema_view.schema, R_NilValue, R_NilValue));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);
  // We do need to set the protected member of the array external pointer to signal that
  // it is not an independent array (i.e., force a shallow copy).
  SEXP array_xptr =
      PROTECT(R_MakeExternalPtr(converter->array_view.array, schema_xptr, converter_xptr));
  Rf_setAttrib(array_xptr, R_ClassSymbol, nanoarrow_cls_array);

  SEXP offset_sexp =
      PROTECT(Rf_ScalarReal(converter->src.array_view->offset + converter->src.offset));
  SEXP length_sexp = PROTECT(Rf_ScalarReal(converter->src.length));

  SEXP fun = PROTECT(Rf_install("convert_fallback_other"));
  SEXP call = PROTECT(
      Rf_lang5(fun, array_xptr, offset_sexp, length_sexp, converter->ptype_view.ptype));
  SEXP result_src = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));

  // Currently this method can only handle the case where result_src and dst have the same
  // SEXP type and length. This won't work for a data frame/record array-like result.
  if (Rf_xlength(result_src) != converter->dst.length) {
    Rf_error("Unexpected length in result of nanoarrow:::convert_fallback_other()");
  }

  if (TYPEOF(result_src) != TYPEOF(converter->dst.vec_sexp)) {
    Rf_error("Unexpected SEXP type in result of nanoarrow:::convert_fallback_other()");
  }

  switch (TYPEOF(result_src)) {
    case REALSXP:
      memcpy(REAL(converter->dst.vec_sexp) + converter->dst.offset, REAL(result_src),
             converter->dst.length * sizeof(double));
      break;
    case INTSXP:
    case LGLSXP:
      memcpy(INTEGER(converter->dst.vec_sexp) + converter->dst.offset,
             INTEGER(result_src), converter->dst.length * sizeof(int));
      break;
    case STRSXP:
      for (R_xlen_t i = 0; i < converter->dst.length; i++) {
        SET_STRING_ELT(converter->dst.vec_sexp, converter->dst.offset + i,
                       STRING_ELT(result_src, i));
      }
      break;
    case VECSXP:
      for (R_xlen_t i = 0; i < converter->dst.length; i++) {
        SET_VECTOR_ELT(converter->dst.vec_sexp, converter->dst.offset + i,
                       VECTOR_ELT(result_src, i));
      }
      break;
    case NILSXP:
      // Do nothing if the function returned NULL
      break;
    default:
      Rf_error(
          "Unhandled SEXP type in conversion of nanoarrow:::convert_fallback_other()");
      break;
  }

  UNPROTECT(7);
  return NANOARROW_OK;
}

static int nanoarrow_materialize_data_frame(struct RConverter* converter,
                                            SEXP converter_xptr) {
  if (converter->ptype_view.vector_type != VECTOR_TYPE_DATA_FRAME) {
    return EINVAL;
  }

  // Make sure we error for dictionary types
  if (converter->src.array_view->array->dictionary != NULL) {
    return EINVAL;
  }

  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);

  switch (converter->array_view.storage_type) {
    case NANOARROW_TYPE_STRUCT:
      for (R_xlen_t i = 0; i < converter->n_children; i++) {
        converter->children[i]->src.offset = converter->src.offset;
        converter->children[i]->src.length = converter->src.length;
        converter->children[i]->dst.offset = converter->dst.offset;
        converter->children[i]->dst.length = converter->dst.length;
        SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
        NANOARROW_RETURN_NOT_OK(
            nanoarrow_materialize(converter->children[i], child_converter_xptr));
      }
      return NANOARROW_OK;

    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      // Pre-fill everything with nulls
      fill_vec_with_nulls(converter->dst.vec_sexp, converter->dst.offset,
                          converter->dst.length);

      // Fill in the possibly non-null values one at a time
      for (R_xlen_t i = 0; i < converter->dst.length; i++) {
        int64_t child_index = ArrowArrayViewUnionChildIndex(&converter->array_view,
                                                            converter->src.offset + i);
        int64_t child_offset = ArrowArrayViewUnionChildOffset(&converter->array_view,
                                                              converter->src.offset + i);
        converter->children[child_index]->src.offset = child_offset;
        converter->children[child_index]->src.length = 1;
        converter->children[child_index]->dst.offset = converter->dst.offset + i;
        converter->children[child_index]->dst.length = 1;
        SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, child_index);
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize(converter->children[child_index],
                                                      child_converter_xptr));
      }
      return NANOARROW_OK;

    default:
      return ENOTSUP;
  }
}

static int materialize_list_element(struct RConverter* converter, SEXP converter_xptr,
                                    int64_t offset, int64_t length) {
  if (nanoarrow_converter_reserve(converter_xptr, length) != NANOARROW_OK) {
    nanoarrow_converter_stop(converter_xptr);
  }

  converter->src.offset = offset;
  converter->src.length = length;
  converter->dst.offset = 0;
  converter->dst.length = length;

  if (nanoarrow_converter_materialize_n(converter_xptr, length) != length) {
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(nanoarrow_converter_finalize(converter_xptr));
  return NANOARROW_OK;
}

static int nanoarrow_materialize_list_of(struct RConverter* converter,
                                         SEXP converter_xptr) {
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);
  struct RConverter* child_converter = converter->children[0];
  SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, 0);

  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;

  // Make sure we error for dictionary types
  if (src->array_view->array->dictionary != NULL) {
    return EINVAL;
  }

  const int32_t* offsets = src->array_view->buffer_views[1].data.as_int32;
  const int64_t* large_offsets = src->array_view->buffer_views[1].data.as_int64;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  int64_t offset;
  int64_t length;

  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return NANOARROW_OK;
    case NANOARROW_TYPE_LIST:
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          offset = offsets[raw_src_offset + i];
          length = offsets[raw_src_offset + i + 1] - offset;
          NANOARROW_RETURN_NOT_OK(materialize_list_element(
              child_converter, child_converter_xptr, offset, length));
          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i,
                         nanoarrow_converter_release_result(child_converter_xptr));
        }
      }
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          offset = large_offsets[raw_src_offset + i];
          length = large_offsets[raw_src_offset + i + 1] - offset;
          NANOARROW_RETURN_NOT_OK(materialize_list_element(
              child_converter, child_converter_xptr, offset, length));
          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i,
                         nanoarrow_converter_release_result(child_converter_xptr));
        }
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      length = src->array_view->layout.child_size_elements;
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          offset = (raw_src_offset + i) * length;
          NANOARROW_RETURN_NOT_OK(materialize_list_element(
              child_converter, child_converter_xptr, offset, length));
          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i,
                         nanoarrow_converter_release_result(child_converter_xptr));
        }
      }
      break;
    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_base(struct RConverter* converter, SEXP converter_xptr) {
  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;
  struct MaterializeOptions* options = converter->options;

  switch (converter->ptype_view.vector_type) {
    case VECTOR_TYPE_UNSPECIFIED:
      return nanoarrow_materialize_unspecified(src, dst, options);
    case VECTOR_TYPE_LGL:
      return nanoarrow_materialize_lgl(src, dst, options);
    case VECTOR_TYPE_INT:
      return nanoarrow_materialize_int(src, dst, options);
    case VECTOR_TYPE_DBL:
      return nanoarrow_materialize_dbl(converter);
    case VECTOR_TYPE_CHR:
      return nanoarrow_materialize_chr(converter);
    case VECTOR_TYPE_POSIXCT:
      return nanoarrow_materialize_posixct(converter);
    case VECTOR_TYPE_DATE:
      return nanoarrow_materialize_date(converter);
    case VECTOR_TYPE_DIFFTIME:
      return nanoarrow_materialize_difftime(converter);
    case VECTOR_TYPE_BLOB:
      return nanoarrow_materialize_blob(src, dst, options);
    case VECTOR_TYPE_LIST_OF:
      return nanoarrow_materialize_list_of(converter, converter_xptr);
    case VECTOR_TYPE_DATA_FRAME:
      return nanoarrow_materialize_data_frame(converter, converter_xptr);
    default:
      return nanoarrow_materialize_other(converter, converter_xptr);
  }
}

int nanoarrow_materialize(struct RConverter* converter, SEXP converter_xptr) {
  int result = nanoarrow_materialize_base(converter, converter_xptr);

  if (result != NANOARROW_OK) {
    return nanoarrow_materialize_other(converter, converter_xptr);
  } else {
    return NANOARROW_OK;
  }
}
