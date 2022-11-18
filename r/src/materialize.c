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

static int nanoarrow_materialize_data_frame(struct RConverter* converter,
                                            SEXP converter_xptr) {
  if (converter->ptype_view.vector_type != VECTOR_TYPE_DATA_FRAME) {
    return EINVAL;
  }

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    converter->children[i]->src.offset = converter->src.offset;
    converter->children[i]->src.length = converter->src.length;
    converter->children[i]->dst.offset = converter->dst.offset;
    converter->children[i]->dst.length = converter->dst.length;
    NANOARROW_RETURN_NOT_OK(
        nanoarrow_materialize(converter->children[i], converter_xptr));
  }

  return NANOARROW_OK;
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

int nanoarrow_materialize(struct RConverter* converter, SEXP converter_xptr) {
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
      return nanoarrow_materialize_chr(src, dst, options);
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
      return ENOTSUP;
  }
}
