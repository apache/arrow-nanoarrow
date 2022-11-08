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
  SEXP result;

  switch (vector_type) {
    case VECTOR_TYPE_LGL:
      result = PROTECT(Rf_allocVector(LGLSXP, len));
      break;
    case VECTOR_TYPE_INT:
      result = PROTECT(Rf_allocVector(INTSXP, len));
      break;
    case VECTOR_TYPE_DBL:
      result = PROTECT(Rf_allocVector(REALSXP, len));
      break;
    case VECTOR_TYPE_CHR:
      result = PROTECT(Rf_allocVector(STRSXP, len));
      break;
    default:
      return R_NilValue;
  }

  UNPROTECT(1);
  return result;
}

int nanoarrow_ptype_is_data_frame(SEXP ptype) {
  return Rf_isObject(ptype) && TYPEOF(ptype) == VECSXP &&
         (Rf_inherits(ptype, "data.frame") ||
          (Rf_xlength(ptype) > 0 && Rf_getAttrib(ptype, R_NamesSymbol) != R_NilValue));
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
        SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
        INTEGER(rownames)[0] = NA_INTEGER;
        INTEGER(rownames)[1] = len;
        Rf_setAttrib(result, R_RowNamesSymbol, rownames);
        UNPROTECT(1);
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

static int nanoarrow_materialize_data_frame(struct RConverter* converter) {
  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    converter->children[i]->src.offset = converter->src.offset;
    converter->children[i]->src.length = converter->src.length;
    converter->children[i]->dst.offset = converter->dst.offset;
    converter->children[i]->dst.length = converter->dst.length;
    NANOARROW_RETURN_NOT_OK(nanoarrow_materialize(converter->children[i]));
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_list_of(struct RConverter* converter) {
  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;

  const int32_t* offsets = src->array_view->buffer_views[1].data.as_int32;
  const int64_t* large_offsets = src->array_view->buffer_views[1].data.as_int64;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  struct RConverter* child_converter = converter->children[0];
  struct ArrayViewSlice* child_src = &child_converter->src;
  struct VectorSlice* child_dst = &child_converter->dst;
  child_dst->offset = 0;

  int convert_result;

  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return NANOARROW_OK;
    case NANOARROW_TYPE_LIST:
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          child_src->offset = offsets[raw_src_offset + i];
          child_src->length = offsets[raw_src_offset + i + 1] - child_src->offset;

          child_dst->vec_sexp = PROTECT(nanoarrow_materialize_realloc(
              child_converter->ptype_view.ptype, child_src->length));
          child_dst->length = child_src->length;
          convert_result = nanoarrow_materialize(child_converter);
          if (convert_result != NANOARROW_OK) {
            UNPROTECT(1);
            return EINVAL;
          }

          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i, child_dst->vec_sexp);
          UNPROTECT(1);
        }
      }
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          child_src->offset = large_offsets[raw_src_offset + i];
          child_src->length = large_offsets[raw_src_offset + i + 1] - child_src->offset;

          child_dst->vec_sexp = PROTECT(nanoarrow_materialize_realloc(
              child_converter->ptype_view.ptype, child_src->length));
          child_dst->length = child_src->length;
          convert_result = nanoarrow_materialize(child_converter);
          if (convert_result != NANOARROW_OK) {
            UNPROTECT(1);
            return EINVAL;
          }

          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i, child_dst->vec_sexp);
          UNPROTECT(1);
        }
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      child_src->length = src->array_view->layout.child_size_elements;
      child_dst->length = child_src->length;
      for (int64_t i = 0; i < dst->length; i++) {
        if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
          child_src->offset = (raw_src_offset + i) * child_src->length;
          child_dst->vec_sexp = PROTECT(nanoarrow_materialize_realloc(
              child_converter->ptype_view.ptype, child_src->length));
          convert_result = nanoarrow_materialize(child_converter);
          if (convert_result != NANOARROW_OK) {
            UNPROTECT(1);
            return EINVAL;
          }

          SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i, child_dst->vec_sexp);
          UNPROTECT(1);
        }
      }
      break;
    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

int nanoarrow_materialize(struct RConverter* converter) {
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
      return nanoarrow_materialize_dbl(src, dst, options);
    // case VECTOR_TYPE_ALTREP_CHR:
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
      return nanoarrow_materialize_list_of(converter);
    case VECTOR_TYPE_DATA_FRAME:
      return nanoarrow_materialize_data_frame(converter);
    default:
      return ENOTSUP;
  }
}
