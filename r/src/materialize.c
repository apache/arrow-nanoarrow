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

SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len) {
  SEXP result;

  if (Rf_isObject(ptype)) {
    if (Rf_inherits(ptype, "data.frame")) {
      R_xlen_t num_cols = Rf_xlength(ptype);
      result = PROTECT(Rf_allocVector(VECSXP, num_cols));
      for (R_xlen_t i = 0; i < num_cols; i++) {
        SET_VECTOR_ELT(result, i,
                       nanoarrow_materialize_realloc(VECTOR_ELT(ptype, i), len));
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

static int nanoarrow_materialize_unspecified(struct ArrayViewSlice* src,
                                             struct VectorSlice* dst,
                                             struct MaterializeOptions* options) {
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

static int nanoarrow_materialize_lgl(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                     struct MaterializeOptions* options) {
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

static int nanoarrow_materialize_int(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                     struct MaterializeOptions* options) {
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
        result[dst->offset + i] =
            ArrowArrayViewGetIntUnsafe(src->array_view, src->offset + i);
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

static int nanoarrow_materialize_dbl(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                     struct MaterializeOptions* options) {
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
        result[dst->offset + i] =
            ArrowArrayViewGetDoubleUnsafe(src->array_view, src->offset + i);
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

static int nanoarrow_materialize_chr(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                     struct MaterializeOptions* options) {
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      break;
    default:
      return EINVAL;
  }

  if (src->array_view->storage_type == NANOARROW_TYPE_NA) {
    for (R_xlen_t i = 0; i < dst->length; i++) {
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
    }

    return NANOARROW_OK;
  }

  struct ArrowStringView item;
  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(src->array_view, src->offset + i);
      SET_STRING_ELT(dst->vec_sexp, dst->offset + i,
                     Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
    }
  }

  return NANOARROW_OK;
}

static int nanoarrow_materialize_blob(struct ArrayViewSlice* src, struct VectorSlice* dst,
                                      struct MaterializeOptions* options) {
  switch (src->array_view->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
      break;
    default:
      return EINVAL;
  }

  if (src->array_view->storage_type == NANOARROW_TYPE_NA) {
    return NANOARROW_OK;
  }

  struct ArrowBufferView item;
  SEXP item_sexp;
  for (R_xlen_t i = 0; i < dst->length; i++) {
    if (!ArrowArrayViewIsNull(src->array_view, src->offset + i)) {
      item = ArrowArrayViewGetBytesUnsafe(src->array_view, src->offset + i);
      item_sexp = PROTECT(Rf_allocVector(RAWSXP, item.n_bytes));
      memcpy(RAW(item_sexp), item.data.data, item.n_bytes);
      SET_VECTOR_ELT(dst->vec_sexp, dst->offset + i, item_sexp);
      UNPROTECT(1);
    }
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

static int nanoarrow_materialize_date(struct RConverter* converter) {
  if (converter->ptype_view.sexp_type == REALSXP) {
    switch (converter->schema_view.data_type) {
      case NANOARROW_TYPE_NA:
      case NANOARROW_TYPE_DATE32:
        return nanoarrow_materialize_dbl(&converter->src, &converter->dst,
                                         converter->options);
      default:
        break;
    }
  }

  return EINVAL;
}

static int nanoarrow_materialize_difftime(struct RConverter* converter) {
  if (converter->ptype_view.sexp_type == REALSXP) {
    switch (converter->schema_view.data_type) {
      case NANOARROW_TYPE_NA:
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(
            &converter->src, &converter->dst, converter->options));
        return NANOARROW_OK;
      case NANOARROW_TYPE_TIME32:
      case NANOARROW_TYPE_TIME64:
      case NANOARROW_TYPE_DURATION:
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(
            &converter->src, &converter->dst, converter->options));
        break;
      default:
        return EINVAL;
    }

    double scale;
    switch (converter->ptype_view.r_time_units) {
      case R_TIME_UNIT_MINUTES:
        scale = 1.0 / 60;
        break;
      case R_TIME_UNIT_HOURS:
        scale = 1.0 / (60 * 60);
        break;
      case R_TIME_UNIT_DAYS:
        scale = 1.0 / (60 * 60 * 24);
        break;
      case R_TIME_UNIT_WEEKS:
        scale = 1.0 / (60 * 60 * 24 * 7);
        break;
      default:
        scale = 1.0;
        break;
    }

    switch (converter->schema_view.time_unit) {
      case NANOARROW_TIME_UNIT_SECOND:
        scale *= 1;
        break;
      case NANOARROW_TIME_UNIT_MILLI:
        scale *= 1e-3;
        break;
      case NANOARROW_TIME_UNIT_MICRO:
        scale *= 1e-6;
        break;
      case NANOARROW_TIME_UNIT_NANO:
        scale *= 1e-9;
        break;
      default:
        return EINVAL;
    }

    if (scale != 1) {
      double* result = REAL(converter->dst.vec_sexp);
      for (int64_t i = 0; i < converter->dst.length; i++) {
        result[converter->dst.offset + i] = result[converter->dst.offset + i] * scale;
      }
    }

    return NANOARROW_OK;
  }

  return EINVAL;
}

static int nanoarrow_materialize_posixct(struct RConverter* converter) {
  if (converter->ptype_view.sexp_type == REALSXP) {
    enum ArrowTimeUnit time_unit;
    switch (converter->schema_view.data_type) {
      case NANOARROW_TYPE_NA:
        time_unit = NANOARROW_TIME_UNIT_SECOND;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(
            &converter->src, &converter->dst, converter->options));
        break;
      case NANOARROW_TYPE_DATE64:
        time_unit = NANOARROW_TIME_UNIT_MILLI;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(
            &converter->src, &converter->dst, converter->options));
        break;
      case NANOARROW_TYPE_TIMESTAMP:
        time_unit = converter->schema_view.time_unit;
        NANOARROW_RETURN_NOT_OK(nanoarrow_materialize_dbl(
            &converter->src, &converter->dst, converter->options));
        break;
      default:
        return EINVAL;
    }

    double scale;
    switch (time_unit) {
      case NANOARROW_TIME_UNIT_SECOND:
        scale = 1;
        break;
      case NANOARROW_TIME_UNIT_MILLI:
        scale = 1e-3;
        break;
      case NANOARROW_TIME_UNIT_MICRO:
        scale = 1e-6;
        break;
      case NANOARROW_TIME_UNIT_NANO:
        scale = 1e-9;
        break;
      default:
        return EINVAL;
    }

    if (scale != 1) {
      double* result = REAL(converter->dst.vec_sexp);
      for (int64_t i = 0; i < converter->dst.length; i++) {
        result[converter->dst.offset + i] = result[converter->dst.offset + i] * scale;
      }
    }

    return NANOARROW_OK;
  }

  return EINVAL;
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
