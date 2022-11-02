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

// Note: These conversions are not currently written for safety rather than
// speed. We could make use of C++ templating to provide faster and/or more
// readable conversions here with a C entry point.

SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len) {
  SEXP result;

  switch (vector_type) {
    case VECTOR_TYPE_UNSPECIFIED:
      result = PROTECT(Rf_allocVector(LGLSXP, len));
      Rf_setAttrib(result, R_ClassSymbol, Rf_mkString("vctrs_unspecified"));
      break;
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

SEXP nanoarrow_alloc_ptype(SEXP ptype, R_xlen_t len) {
  SEXP result;

  if (Rf_isObject(ptype)) {
    if (Rf_inherits(ptype, "data.frame")) {
      R_xlen_t num_cols = Rf_xlength(ptype);
      result = PROTECT(Rf_allocVector(VECSXP, num_cols));
      for (R_xlen_t i = 0; i < num_cols; i++) {
        SET_VECTOR_ELT(result, i, nanoarrow_alloc_ptype(VECTOR_ELT(ptype, i), len));
      }

      // Set attributes from ptype
      Rf_setAttrib(result, R_NamesSymbol, Rf_getAttrib(ptype, R_NamesSymbol));
      Rf_setAttrib(result, R_ClassSymbol, Rf_getAttrib(ptype, R_ClassSymbol));

      // ...except rownames
      SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
      INTEGER(rownames)[0] = NA_INTEGER;
      INTEGER(rownames)[1] = len;
      Rf_setAttrib(result, R_RowNamesSymbol, rownames);
    } else if (Rf_inherits(ptype, "matrix")) {
      result = PROTECT(Rf_allocMatrix(TYPEOF(ptype), len, Rf_ncols(ptype)));
      Rf_copyMostAttrib(ptype, result);
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

static int nanoarrow_materialize_legacy(struct ArrowArrayView* array_view,
                                        SEXP result_sexp) {
  struct ArrayViewSlice src = DefaultArrayViewSlice(array_view);
  struct VectorSlice dst = DefaultVectorSlice(result_sexp);
  struct MaterializeOptions options = DefaultMaterializeOptions();
  struct MaterializeContext context = DefaultMaterializeContext();

  return nanoarrow_materialize(&src, &dst, &options, &context);
}

SEXP nanoarrow_materialize_unspecified(struct ArrowArrayView* array_view) {
  SEXP result_sexp =
      PROTECT(nanoarrow_alloc_type(VECTOR_TYPE_UNSPECIFIED, array_view->array->length));

  if (nanoarrow_materialize_legacy(array_view, result_sexp)) {
    UNPROTECT(1);
    return R_NilValue;
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_lgl(struct ArrowArrayView* array_view) {
  SEXP result_sexp =
      PROTECT(nanoarrow_alloc_type(VECTOR_TYPE_LGL, array_view->array->length));

  if (nanoarrow_materialize_legacy(array_view, result_sexp)) {
    UNPROTECT(1);
    return R_NilValue;
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_int(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(INTSXP, array_view->array->length));

  if (nanoarrow_materialize_legacy(array_view, result_sexp)) {
    UNPROTECT(1);
    return R_NilValue;
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_dbl(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(REALSXP, array_view->array->length));

  if (nanoarrow_materialize_legacy(array_view, result_sexp)) {
    UNPROTECT(1);
    return R_NilValue;
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_chr(struct ArrowArrayView* array_view) {
  SEXP result_sexp = PROTECT(Rf_allocVector(STRSXP, array_view->array->length));

  if (array_view->storage_type == NANOARROW_TYPE_NA) {
    for (R_xlen_t i = 0; i < array_view->array->length; i++) {
      SET_STRING_ELT(result_sexp, i, NA_STRING);
    }

    UNPROTECT(1);
    return result_sexp;
  }

  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < array_view->array->length; i++) {
    if (ArrowArrayViewIsNull(array_view, i)) {
      SET_STRING_ELT(result_sexp, i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(array_view, i);
      SET_STRING_ELT(result_sexp, i, Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
    }
  }

  UNPROTECT(1);
  return result_sexp;
}

SEXP nanoarrow_materialize_list_of_raw(struct ArrowArrayView* array_view) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      break;
    default:
      return R_NilValue;
  }

  SEXP result_sexp = PROTECT(Rf_allocVector(VECSXP, array_view->array->length));

  if (array_view->storage_type == NANOARROW_TYPE_NA) {
    UNPROTECT(1);
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

  UNPROTECT(1);
  return result_sexp;
}

static int nanoarrow_materialize_data_frame(struct ArrayViewSlice* src,
                                            struct VectorSlice* dst,
                                            struct MaterializeOptions* options,
                                            struct MaterializeContext* context) {
  if (src->array_view->storage_type != NANOARROW_TYPE_STRUCT) {
    return EINVAL;
  }

  if (src->array_view->n_children != Rf_xlength(dst->vec_sexp)) {
    return EINVAL;
  }

  struct ArrayViewSlice src_child = *src;
  struct VectorSlice dst_child = *dst;
  struct MaterializeContext child_context;
  SEXP names = Rf_getAttrib(dst_child.vec_sexp, R_NamesSymbol);

  for (int64_t i = 0; i < src->array_view->n_children; i++) {
    child_context.context = Rf_translateCharUTF8(STRING_ELT(names, i));
    src_child.array_view = src->array_view->children[i];
    dst_child.vec_sexp = VECTOR_ELT(dst->vec_sexp, i);
    NANOARROW_RETURN_NOT_OK(
        nanoarrow_materialize(&src_child, &dst_child, options, &child_context));
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_matrix(struct ArrayViewSlice* src,
                                        struct VectorSlice* dst,
                                        struct MaterializeOptions* options,
                                        struct MaterializeContext* context) {
  Rf_error("Materialize to matrix not implemented");
}

static int nanoarrow_materialize_unspecified2(struct ArrayViewSlice* src,
                                              struct VectorSlice* dst,
                                              struct MaterializeOptions* options,
                                              struct MaterializeContext* context) {
  int* result = LOGICAL(dst->vec_sexp);

  int64_t total_offset = src->array_view->array->offset + src->offset;
  int64_t length = src->length;
  const uint8_t* bits = src->array_view->buffer_views[0].data.as_uint8;

  if (length == 0 || src->array_view->storage_type == NANOARROW_TYPE_NA ||
      ArrowBitCountSet(bits, total_offset, length) == 0) {
    // We can blindly set all the values to NA_LOGICAL without checking
    for (int64_t i = 0; i < length; i++) {
      result[dst->offset + i] = NA_LOGICAL;
    }
  } else {
    // Count non-null values and warn
    int64_t n_bad_values = 0;
    for (int64_t i = 0; i < length; i++) {
      n_bad_values += ArrowBitGet(bits, total_offset + i);
      result[dst->offset + i] = NA_LOGICAL;
    }

    if (n_bad_values > 0) {
      Rf_warning("%ld non-null value(s) set to NA", (long)n_bad_values);
    }
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_lgl2(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                      struct MaterializeOptions* options,
                                      struct MaterializeContext* context) {
  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  const uint8_t* data_buffer = src->array_view->buffer_views[1].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;
  int* result = LOGICAL(dst->vec_sexp);

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_LOGICAL;
      }
      break;
    case NANOARROW_TYPE_BOOL:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = ArrowBitGet(data_buffer, src->offset + i);
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_LOGICAL;
          }
        }
      }
      break;
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      for (R_xlen_t i = 0; i < src->array_view->array->length; i++) {
        result[dst->offset + i] =
            ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i) != 0;
      }

      // Set any nulls to NA_LOGICAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_LOGICAL;
          }
        }
      }
      break;

    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_int2(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                      struct MaterializeOptions* options,
                                      struct MaterializeContext* context) {
  int* result = INTEGER(dst->vec_sexp);
  int64_t n_bad_values = 0;

  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_INTEGER;
      }
      break;
    case NANOARROW_TYPE_INT32:
      memcpy(result + dst->offset,
             src->array_view->buffer_views[1].data.as_int32 + raw_src_offset,
             dst->length * sizeof(int32_t));

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_INTEGER;
          }
        }
      }
      break;
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
      // No need to bounds check for these types
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
      }

      // Set any nulls to NA_INTEGER
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_INTEGER;
          }
        }
      }
      break;
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      // Loop + bounds check. Because we don't know what memory might be
      // in a null slot, we have to check nulls if there are any.
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (ArrowBitGet(is_valid, raw_src_offset + i)) {
            int64_t value = ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
            if (value > INT_MAX || value <= NA_INTEGER) {
              result[dst->offset + i] = NA_INTEGER;
              n_bad_values++;
            } else {
              result[dst->offset + i] = value;
            }
          } else {
            result[dst->offset + i] = NA_INTEGER;
          }
        }
      } else {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          int64_t value = ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
          if (value > INT_MAX || value <= NA_INTEGER) {
            result[dst->offset + i] = NA_INTEGER;
            n_bad_values++;
          } else {
            result[dst->offset + i] = value;
          }
        }
      }
      break;

    default:
      return EINVAL;
  }

  if (n_bad_values > 0) {
    Rf_warning("%ld value(s) outside integer range set to NA", (long)n_bad_values);
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_dbl2(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                      struct MaterializeOptions* options,
                                      struct MaterializeContext* context) {
  double* result = REAL(dst->vec_sexp);

  // True for all the types supported here
  const uint8_t* is_valid = src->array_view->buffer_views[0].data.as_uint8;
  int64_t raw_src_offset = src->array_view->array->offset + src->offset;

  // Fill the buffer
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = NA_REAL;
      }
      break;
    case NANOARROW_TYPE_DOUBLE:
      memcpy(result + dst->offset,
             src->array_view->buffer_views[1].data.as_double + raw_src_offset,
             dst->length * sizeof(double));

      // Set any nulls to NA_REAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_REAL;
          }
        }
      }
      break;
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
      // TODO: implement bounds check for int64 and uint64, but instead
      // of setting to NA, just warn (because sequential values might not
      // roundtrip above 2^51 ish)
      for (R_xlen_t i = 0; i < dst->length; i++) {
        result[dst->offset + i] = ArrowArrayViewGetDoubleUnsafe(src->array_view, src->offset + i);
      }

      // Set any nulls to NA_REAL
      if (is_valid != NULL && src->array_view->array->null_count != 0) {
        for (R_xlen_t i = 0; i < dst->length; i++) {
          if (!ArrowBitGet(is_valid, raw_src_offset + i)) {
            result[dst->offset + i] = NA_REAL;
          }
        }
      }
      break;

    default:
      return EINVAL;
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_chr2(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                      struct MaterializeOptions* options,
                                      struct MaterializeContext* context) {
  Rf_error("Materialize to chr not implemented");
}

int nanoarrow_materialize(struct ArrayViewSlice* src, struct VectorSlice* dst,
                          struct MaterializeOptions* options,
                          struct MaterializeContext* context) {
  if (src->length != dst->length) {
    Rf_error(
        "Can't materialize ArrayViewSlice with length %ld into VectorSlice of length %ld",
        (long)src->length, (long)dst->length);
  }

  // Skip it all if there is no materializing to do
  if (src->length == 0) {
    return NANOARROW_OK;
  }

  if (Rf_isObject(dst->vec_sexp)) {
    if (Rf_inherits(dst->vec_sexp, "data.frame")) {
      return nanoarrow_materialize_data_frame(src, dst, options, context);
    } else if (Rf_inherits(dst->vec_sexp, "matrix")) {
      return nanoarrow_materialize_matrix(src, dst, options, context);
    } else if (Rf_inherits(dst->vec_sexp, "vctrs_unspecified")) {
      return nanoarrow_materialize_unspecified2(src, dst, options, context);
    }
  }

  switch (TYPEOF(dst->vec_sexp)) {
    case LGLSXP:
      return nanoarrow_materialize_lgl2(src, dst, options, context);
    case INTSXP:
      return nanoarrow_materialize_int2(src, dst, options, context);
    case REALSXP:
      return nanoarrow_materialize_dbl2(src, dst, options, context);
    case STRSXP:
      return nanoarrow_materialize_chr2(src, dst, options, context);
    default:
      break;
  }

  return EINVAL;
}
