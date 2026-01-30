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
#include <Rversion.h>

#include "array.h"
#include "nanoarrow.h"
#include "nanoarrow/r.h"
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
#include "materialize_int64.h"
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
#if R_VERSION >= R_Version(4, 6, 0)
  return (int)R_hasAttrib(x, sym);
#else
  for (SEXP atts = ATTRIB(x); atts != R_NilValue; atts = CDR(atts)) {
    if (TAG(atts) == sym) return TRUE;
  }
  return FALSE;
#endif
}

R_xlen_t nanoarrow_data_frame_size(SEXP x) {
  // If this is actually a data frame, check the rownames. ALTREP ensures
  // that "expanding" these rownames by accessing the attribute does not
  // actually compute millions of strings.
  if (Rf_inherits(x, "data.frame")) {
    return Rf_xlength(Rf_getAttrib(x, R_RowNamesSymbol));
  }

  // If we are here, this is a struct style vctr. If we have a plain vctr
  // as the first element we can just compute its length from the Rf_xlength()
  // of the first column.
  if (Rf_length(x) > 0 && !Rf_isObject(VECTOR_ELT(x, 0))) {
    return Rf_xlength(VECTOR_ELT(x, 0));
  } else {
    // Otherwise, we need S3 dispatch to compute the length for us.
    SEXP length_sym = PROTECT(Rf_install("length"));
    SEXP call_sexp = PROTECT(Rf_lang2(length_sym, x));
    SEXP length_sexp = PROTECT(Rf_eval(call_sexp, R_BaseEnv));
    if (Rf_xlength(length_sexp) != 1) {
      Rf_error("length() return value with size != 1");
    }

    R_xlen_t out = 0;
    switch (TYPEOF(length_sexp)) {
      case INTSXP:
        out = INTEGER(length_sexp)[0];
        break;
      case REALSXP:
        out = (R_xlen_t)REAL(length_sexp)[0];
        break;
      default:
        Rf_error("length() return value with unexpected type");
    }

    UNPROTECT(3);
    return out;
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
    INTEGER(rownames)[1] = (int)(-len);
    Rf_setAttrib(x, R_RowNamesSymbol, rownames);
    UNPROTECT(1);
  } else {
    SEXP length_dbl = PROTECT(Rf_ScalarReal((double)len));
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

int nanoarrow_ptype_is_nanoarrow_vctr(SEXP ptype) {
  return Rf_inherits(ptype, "nanoarrow_vctr");
}

SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len) {
  SEXP result;

  if (Rf_isMatrix(ptype)) {
    // The actual value is built in the child converter but we can't have
    // a NULL here because that confuses the internals into thinking that
    // the allocate was never called.
    result = PROTECT(Rf_allocVector(TYPEOF(ptype), 0));
  } else if (Rf_isObject(ptype)) {
    // There may be a more accurate test that more precisely captures the case
    // where a user has specified a valid ptype that doesn't work in a preallocate
    // + fill conversion.
    if (Rf_inherits(ptype, "factor")) {
      SEXP levels = Rf_getAttrib(ptype, R_LevelsSymbol);
      if (Rf_length(levels) == 0) {
        Rf_error("Can't allocate ptype of class 'factor' with empty levels");
      }
    }

    if (nanoarrow_ptype_is_nanoarrow_vctr(ptype)) {
      // The object we return here is one that will accumulate chunks and
      // be finalized with a value (rather than being strictly copied into
      // after every new chunk is seen).
      result = PROTECT(Rf_allocVector(INTSXP, len));
      Rf_copyMostAttrib(ptype, result);

      // For the purposes of building the list of chunks, chunks is a pairlist
      // (it will be converted to a regular list when this converter is finalized)
      // Technically the first value here won't be used (this simplifies the
      // appending).
      SEXP chunks_list = PROTECT(Rf_list1(R_NilValue));

      // To start, the chunks list and the end of the chunks list are the same node
      SEXP chunks_tail_sym = PROTECT(Rf_install("chunks_tail"));
      SEXP chunks_sym = PROTECT(Rf_install("chunks"));
      Rf_setAttrib(result, chunks_sym, chunks_list);
      Rf_setAttrib(result, chunks_tail_sym, chunks_list);

      UNPROTECT(3);
    } else if (nanoarrow_ptype_is_data_frame(ptype)) {
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
    case RAWSXP:
      // Not perfect: raw() doesn't really support NA in R
      memset(RAW(x), 0, len * sizeof(char));
      break;
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
    case CPLXSXP: {
      Rcomplex* values = COMPLEX(x);
      Rcomplex na_value;
      na_value.r = NA_REAL;
      na_value.i = NA_REAL;

      for (R_xlen_t i = 0; i < len; i++) {
        values[offset + i] = na_value;
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

static void copy_vec_into(SEXP x, SEXP dst, R_xlen_t offset, R_xlen_t len) {
  if (nanoarrow_ptype_is_nanoarrow_vctr(dst)) {
    Rf_error("Can't copy_vec_into() to nanoarrow_vctr");
  }

  if (nanoarrow_ptype_is_data_frame(dst)) {
    if (!nanoarrow_ptype_is_data_frame(x)) {
      Rf_error("Expected record-style vctr result but got non-record-style result");
    }

    R_xlen_t x_len = nanoarrow_data_frame_size(x);
    if (len != x_len) {
      Rf_error("Unexpected data.frame row count in copy_vec_into()");
    }

    // This does not currently consider column names (i.e., it blindly copies
    // by index).
    if (Rf_xlength(x) != Rf_xlength(dst)) {
      Rf_error("Unexpected data.frame column count in copy_vec_into()");
    }

    for (R_xlen_t i = 0; i < Rf_xlength(x); i++) {
      copy_vec_into(VECTOR_ELT(x, i), VECTOR_ELT(dst, i), offset, len);
    }

    return;
  } else if (nanoarrow_ptype_is_data_frame(x)) {
    Rf_error("Expected non-record-style vctr result but got record-style result");
  }

  if (TYPEOF(dst) != TYPEOF(x)) {
    Rf_error("Unexpected SEXP type in result copy_vec_into()");
  }

  if (Rf_length(x) != len) {
    Rf_error("Unexpected length of result in copy_vec_into()");
  }

  switch (TYPEOF(dst)) {
    case RAWSXP:
      memcpy(RAW(dst) + offset, RAW(x), len * sizeof(uint8_t));
      break;
    case REALSXP:
      memcpy(REAL(dst) + offset, REAL(x), len * sizeof(double));
      break;
    case INTSXP:
    case LGLSXP:
      memcpy(INTEGER(dst) + offset, INTEGER(x), len * sizeof(int));
      break;
    case CPLXSXP:
      memcpy(COMPLEX(dst) + offset, COMPLEX(x), len * sizeof(Rcomplex));
      break;
    case STRSXP:
      for (R_xlen_t i = 0; i < len; i++) {
        SET_STRING_ELT(dst, offset + i, STRING_ELT(x, i));
      }
      break;
    case VECSXP:
      for (R_xlen_t i = 0; i < len; i++) {
        SET_VECTOR_ELT(dst, offset + i, VECTOR_ELT(x, i));
      }
      break;
    default:
      Rf_error("Unhandled SEXP type in copy_vec_into()");
      break;
  }
}

int nanoarrow_materialize_finalize_result(SEXP converter_xptr) {
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP result = VECTOR_ELT(converter_shelter, 4);

  // Materialize never called (e.g., empty stream)
  if (result == R_NilValue) {
    nanoarrow_converter_reserve(converter_xptr, 0);
    result = VECTOR_ELT(converter_shelter, 4);
  }

  if (nanoarrow_ptype_is_nanoarrow_vctr(result)) {
    // Get the schema for this converter. Technically this will overwrite
    // a schema that was provided explicitly; however, we currently do not
    // handle that case.

    SEXP schema_xptr = VECTOR_ELT(converter_shelter, 1);

    // We no longer need to keep track of chunks_tail
    SEXP chunks_tail_sym = PROTECT(Rf_install("chunks_tail"));
    Rf_setAttrib(result, chunks_tail_sym, R_NilValue);

    // We also want to pass on the class of the ptype we received
    SEXP subclass_sexp = Rf_getAttrib(result, R_ClassSymbol);

    // We no longer need the first element of the pairlist, which was
    // intentionally set to R_NilValue.
    SEXP chunks_sym = PROTECT(Rf_install("chunks"));
    SEXP chunks_pairlist0 = Rf_getAttrib(result, chunks_sym);

    // If there were zero chunks, there will be no "first" node
    SEXP chunks_list;
    if (CDR(chunks_pairlist0) == R_NilValue) {
      chunks_list = PROTECT(Rf_allocVector(VECSXP, 0));
    } else {
      chunks_list = PROTECT(Rf_PairToVectorList(CDR(chunks_pairlist0)));
    }

    // Set up the call to new_nanoarrow_vctr
    SEXP new_nanoarrow_vctr_sym = PROTECT(Rf_install("new_nanoarrow_vctr"));
    SEXP new_nanoarrow_vctr_call = PROTECT(
        Rf_lang4(new_nanoarrow_vctr_sym, chunks_list, schema_xptr, subclass_sexp));
    SEXP final_result = PROTECT(Rf_eval(new_nanoarrow_vctr_call, nanoarrow_ns_pkg));

    SET_VECTOR_ELT(converter_shelter, 4, final_result);
    UNPROTECT(6);
  } else if (nanoarrow_ptype_is_data_frame(result)) {
    // For each child, finalize the result and then reassign it
    SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);
    for (R_xlen_t i = 0; i < Rf_xlength(child_converter_xptrs); i++) {
      SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
      NANOARROW_RETURN_NOT_OK(
          nanoarrow_materialize_finalize_result(child_converter_xptr));

      SEXP child_result =
          PROTECT(nanoarrow_converter_release_result(child_converter_xptr));
      SET_VECTOR_ELT(result, i, child_result);
      UNPROTECT(1);
    }
  } else if (converter->ptype_view.vector_type == VECTOR_TYPE_MATRIX) {
    SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);
    SEXP item_converter_xptr = VECTOR_ELT(child_converter_xptrs, 0);
    NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_finalize_result(item_converter_xptr));
    SEXP item_result = PROTECT(nanoarrow_converter_release_result(item_converter_xptr));

    SEXP matrix_symbol = PROTECT(Rf_install("matrix"));
    SEXP nrow_sexp = PROTECT(
        Rf_ScalarInteger(Rf_xlength(item_result) / converter->schema_view.fixed_size));
    SEXP ncol_sexp = PROTECT(Rf_ScalarInteger(converter->schema_view.fixed_size));
    SEXP byrow_sexp = PROTECT(Rf_ScalarLogical(TRUE));
    SEXP matrix_call =
        PROTECT(Rf_lang5(matrix_symbol, item_result, nrow_sexp, ncol_sexp, byrow_sexp));
    SEXP final_result = PROTECT(Rf_eval(matrix_call, R_BaseNamespace));
    SET_VECTOR_ELT(converter_shelter, 4, final_result);
    UNPROTECT(7);
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_nanoarrow_vctr(struct RConverter* converter,
                                                SEXP converter_xptr) {
  // This is a case where the callee needs ownership, which we can do via a
  // shallow copy.
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP schema_xptr = VECTOR_ELT(converter_shelter, 1);
  SEXP array_xptr = VECTOR_ELT(converter_shelter, 2);

  SEXP array_out_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* out_array = nanoarrow_output_array_from_xptr(array_out_xptr);
  array_export(array_xptr, out_array);
  R_SetExternalPtrTag(array_out_xptr, schema_xptr);

  // Update the offset/length in case a slice is being requested from the
  // converter.
  out_array->offset += converter->src.offset;
  out_array->length = converter->src.length;

  // Get the cached copy of the pairlist node at the end of the current
  // chunks list.
  SEXP chunks_tail_sym = PROTECT(Rf_install("chunks_tail"));
  SEXP chunks_tail = PROTECT(Rf_getAttrib(converter->dst.vec_sexp, chunks_tail_sym));

  // Create a length-1 pairlist node containing the chunk
  SEXP next_sexp = PROTECT(Rf_cons(array_out_xptr, R_NilValue));

  // Append it to the end of the current pairlist
  SETCDR(chunks_tail, next_sexp);
  UNPROTECT(1);

  // Update the cached copy of the pairlist node at the end of the current
  // chunks list.
  Rf_setAttrib(converter->dst.vec_sexp, chunks_tail_sym, next_sexp);
  UNPROTECT(3);

  return NANOARROW_OK;
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

  // Special-case the nanoarrow_vctr conversion
  if (Rf_inherits(converter->dst.vec_sexp, "nanoarrow_vctr")) {
    return nanoarrow_materialize_nanoarrow_vctr(converter, converter_xptr);
  }

  // We've ensured proper ownership of array_xptr and ensured that its
  // schema is set, so we can pass these safely to the R-level
  // convert_fallback_other.
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP array_xptr = VECTOR_ELT(converter_shelter, 2);

  // The R code in convert_fallback_other() takes care of ensuring an independent copy
  // with the correct offset/length if it is necessary to update them.
  SEXP offset_sexp = PROTECT(
      Rf_ScalarReal((double)(converter->src.array_view->offset + converter->src.offset)));
  SEXP length_sexp = PROTECT(Rf_ScalarReal((double)converter->src.length));

  SEXP fun = PROTECT(Rf_install("convert_fallback_other"));
  SEXP call = PROTECT(
      Rf_lang5(fun, array_xptr, offset_sexp, length_sexp, converter->ptype_view.ptype));
  SEXP result_src = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));

  // Copy the result into a slice of dst
  copy_vec_into(result_src, converter->dst.vec_sexp, converter->dst.offset,
                converter->dst.length);

  UNPROTECT(5);
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
  nanoarrow_converter_reserve(converter_xptr, length);

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
    case NANOARROW_TYPE_MAP:
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

static int nanoarrow_materialize_matrix(struct RConverter* converter,
                                        SEXP converter_xptr) {
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);
  struct RConverter* child_converter = converter->children[0];
  SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, 0);

  struct ArrayViewSlice* src = &converter->src;

  // Make sure we error for dictionary types
  if (src->array_view->array->dictionary != NULL) {
    return EINVAL;
  }

  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      break;
    default:
      return EINVAL;
  }

  int64_t raw_src_offset = src->array_view->offset + src->offset;
  int64_t list_length = src->array_view->layout.child_size_elements;
  int64_t child_length = list_length * src->length;

  if (list_length != Rf_ncols(converter->ptype_view.ptype)) {
    Rf_error("Can't convert fixed_size_list(list_size=%d) to matrix with %d cols",
             (int)list_length, Rf_ncols(converter->ptype_view.ptype));
  }

  // First, we update the child array offset to account for the parent offset and
  // materialize the child array.
  child_converter->src.offset += raw_src_offset * list_length;
  child_converter->src.length = child_length;
  if (nanoarrow_converter_materialize_n(child_converter_xptr, child_length) !=
      child_length) {
    return EINVAL;
  }

  // If we have parent nulls, we have to project them into the destination
  if (src->array_view->null_count != 0 &&
      src->array_view->buffer_views[0].data.data != NULL) {
    // Here, dst.offset has already been incremented such that it's at the end
    // of the chunk, but we need the original one for fill_vec_with_nulls().
    int64_t original_dst_offset = child_converter->dst.offset - child_length;
    for (int64_t i = 0; i < src->length; i++) {
      if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
        fill_vec_with_nulls(child_converter->dst.vec_sexp,
                            original_dst_offset + (i * list_length), list_length);
      }
    }
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_base(struct RConverter* converter, SEXP converter_xptr) {
  struct ArrayViewSlice* src = &converter->src;
  struct VectorSlice* dst = &converter->dst;
  struct MaterializeOptions* options = converter->options;

  // Make sure extension conversion calls into R
  if (converter->schema_view.extension_name.size_bytes > 0) {
    return nanoarrow_materialize_other(converter, converter_xptr);
  }

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
    case VECTOR_TYPE_INTEGER64:
      return nanoarrow_materialize_int64(src, dst, options);
    case VECTOR_TYPE_BLOB:
      return nanoarrow_materialize_blob(src, dst, options);
    case VECTOR_TYPE_LIST_OF:
      return nanoarrow_materialize_list_of(converter, converter_xptr);
    case VECTOR_TYPE_MATRIX:
      return nanoarrow_materialize_matrix(converter, converter_xptr);
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
