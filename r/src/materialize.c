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

#include "array.h"
#include "materialize.h"
#include "schema.h"

int nanoarrow_materialize(struct RConverter* converter);

R_xlen_t nanoarrow_vec_size(SEXP vec_sexp) {
  if (Rf_isObject(vec_sexp)) {
    if (Rf_inherits(vec_sexp, "data.frame") && Rf_length(vec_sexp) > 0) {
      // Avoid materializing the row.names if we can
      return Rf_xlength(VECTOR_ELT(vec_sexp, 0));
    } else if (Rf_inherits(vec_sexp, "data.frame")) {
      return Rf_xlength(Rf_getAttrib(vec_sexp, R_RowNamesSymbol));
    } else if (Rf_inherits(vec_sexp, "matrix")) {
      return Rf_nrows(vec_sexp);
    }
  }

  return Rf_xlength(vec_sexp);
}

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

static void finalize_converter(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  if (converter != NULL) {
    ArrowArrayViewReset(&converter->array_view);

    if (converter->children != NULL) {
      ArrowFree(converter->children);
    }

    ArrowFree(converter);
  }
}

SEXP nanoarrow_converter_from_type(enum VectorType vector_type) {
  struct RConverter* converter =
      (struct RConverter*)ArrowMalloc(sizeof(struct RConverter));
  if (converter == NULL) {
    Rf_error("Failed to allocate RConverter");
  }

  // 0: ptype, 1: schema_xptr, 2: array_xptr, 3: children, 4: result
  SEXP converter_shelter = PROTECT(Rf_allocVector(VECSXP, 5));
  SEXP converter_xptr =
      PROTECT(R_MakeExternalPtr(converter, R_NilValue, converter_shelter));
  R_RegisterCFinalizer(converter_xptr, &finalize_converter);

  ArrowArrayViewInit(&converter->array_view, NANOARROW_TYPE_UNINITIALIZED);
  converter->schema_view.data_type = NANOARROW_TYPE_UNINITIALIZED;
  converter->schema_view.storage_data_type = NANOARROW_TYPE_UNINITIALIZED;
  converter->src.array_view = &converter->array_view;
  converter->dst.vec_sexp = R_NilValue;
  converter->options = NULL;
  converter->error.message[0] = '\0';
  converter->size = 0;
  converter->capacity = 0;
  converter->n_children = 0;
  converter->children = NULL;

  converter->ptype_view.vector_type = vector_type;
  converter->ptype_view.ptype = R_NilValue;

  switch (vector_type) {
    case VECTOR_TYPE_NULL:
      converter->ptype_view.sexp_type = NILSXP;
      break;
    case VECTOR_TYPE_LGL:
      converter->ptype_view.sexp_type = LGLSXP;
      break;
    case VECTOR_TYPE_INT:
      converter->ptype_view.sexp_type = INTSXP;
      break;
    case VECTOR_TYPE_DBL:
      converter->ptype_view.sexp_type = REALSXP;
      break;
    case VECTOR_TYPE_CHR:
      converter->ptype_view.sexp_type = STRSXP;
      break;
    default:
      UNPROTECT(2);
      return R_NilValue;
  }

  UNPROTECT(2);
  return converter_xptr;
}

static enum RTimeUnits time_units_from_difftime(SEXP ptype) {
  SEXP units_attr = Rf_getAttrib(ptype, Rf_install("units"));
  if (units_attr == R_NilValue || TYPEOF(units_attr) != STRSXP ||
      Rf_length(units_attr) != 1) {
    Rf_error("Expected difftime 'units' attribute of type character(1)");
  }

  const char* dst_units = Rf_translateCharUTF8(STRING_ELT(units_attr, 0));
  if (strcmp(dst_units, "secs") == 0) {
    return R_TIME_UNIT_SECONDS;
  } else if (strcmp(dst_units, "mins") == 0) {
    return R_TIME_UNIT_MINUTES;
  } else if (strcmp(dst_units, "hours") == 0) {
    return R_TIME_UNIT_HOURS;
  } else if (strcmp(dst_units, "days") == 0) {
    return R_TIME_UNIT_DAYS;
  } else if (strcmp(dst_units, "weeks") == 0) {
    return R_TIME_UNIT_WEEKS;
  } else {
    Rf_error("Unexpected value for difftime 'units' attribute");
    return R_TIME_UNIT_SECONDS;
  }
}

static void set_converter_list_of(SEXP converter_xptr, struct RConverter* converter,
                                  SEXP ptype) {
  SEXP child_ptype = Rf_getAttrib(ptype, Rf_install("ptype"));
  if (child_ptype == R_NilValue) {
    Rf_error("Expected attribute 'ptype' for conversion to list_of");
  }

  converter->children = (struct RConverter**)ArrowMalloc(1 * sizeof(struct RConverter*));
  if (converter->children == NULL) {
    Rf_error("Failed to allocate converter children array");
  }
  converter->n_children = 1;

  SEXP child_converter_xptrs = PROTECT(Rf_allocVector(VECSXP, 1));
  SEXP child_converter = PROTECT(nanoarrow_converter_from_ptype(child_ptype));
  converter->children[0] = (struct RConverter*)R_ExternalPtrAddr(child_converter);
  SET_VECTOR_ELT(child_converter_xptrs, 0, child_converter);

  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SET_VECTOR_ELT(converter_shelter, 3, child_converter_xptrs);
  UNPROTECT(2);
}

static int set_converter_children_schema(SEXP converter_xptr, SEXP schema_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  if (schema->n_children != converter->n_children) {
    ArrowErrorSet(&converter->error,
                  "Expected schema with %ld children but got schema with %ld children",
                  (long)converter->n_children, (long)schema->n_children);
    return EINVAL;
  }

  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
    SEXP child_schema_xptr = PROTECT(borrow_schema_child_xptr(schema_xptr, i));
    int result = nanoarrow_converter_set_schema(child_converter_xptr, child_schema_xptr);
    UNPROTECT(1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  return NANOARROW_OK;
}

static int set_converter_children_array(SEXP converter_xptr, SEXP array_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowArray* array = array_from_xptr(array_xptr);

  if (array->n_children != converter->n_children) {
    ArrowErrorSet(&converter->error,
                  "Expected array with %ld children but got array with %ld children",
                  (long)converter->n_children, (long)array->n_children);
    return EINVAL;
  }

  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
    SEXP child_array_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
    int result = nanoarrow_converter_set_array(child_converter_xptr, child_array_xptr);
    UNPROTECT(1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  return NANOARROW_OK;
}

SEXP nanoarrow_converter_from_ptype(SEXP ptype) {
  SEXP converter_xptr = PROTECT(nanoarrow_converter_from_type(VECTOR_TYPE_NULL));
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);

  if (Rf_isObject(ptype)) {
    if (Rf_inherits(ptype, "data.frame")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DATA_FRAME;
      Rf_error("data.frame currently not supported by converter");
    } else if (Rf_inherits(ptype, "blob")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_BLOB;
    } else if (Rf_inherits(ptype, "vctrs_list_of")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_LIST_OF;
      set_converter_list_of(converter_xptr, converter, ptype);
    } else if (Rf_inherits(ptype, "vctrs_unspecified")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_UNSPECIFIED;
    } else if (Rf_inherits(ptype, "Date")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DATE;
      converter->ptype_view.r_time_units = R_TIME_UNIT_DAYS;
    } else if (Rf_inherits(ptype, "POSIXct")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_POSIXCT;
      converter->ptype_view.r_time_units = R_TIME_UNIT_SECONDS;
    } else if (Rf_inherits(ptype, "difftime")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DIFFTIME;
      converter->ptype_view.r_time_units = time_units_from_difftime(ptype);
    } else {
      converter->ptype_view.vector_type = VECTOR_TYPE_OTHER;
    }
  } else {
    switch (TYPEOF(ptype)) {
      case LGLSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_LGL;
        break;
      case INTSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_INT;
        break;
      case REALSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_DBL;
        break;
      case STRSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_CHR;
        break;
      default:
        converter->ptype_view.vector_type = VECTOR_TYPE_OTHER;
        break;
    }
  }

  converter->ptype_view.ptype = ptype;
  converter->ptype_view.sexp_type = TYPEOF(ptype);
  SET_VECTOR_ELT(converter_shelter, 0, ptype);

  UNPROTECT(1);
  return converter_xptr;
}

int nanoarrow_converter_set_schema(SEXP converter_xptr, SEXP schema_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  NANOARROW_RETURN_NOT_OK(
      ArrowSchemaViewInit(&converter->schema_view, schema, &converter->error));
  SET_VECTOR_ELT(converter_shelter, 1, schema_xptr);

  ArrowArrayViewReset(&converter->array_view);
  SET_VECTOR_ELT(converter_shelter, 2, R_NilValue);
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&converter->array_view, schema, &converter->error));

  if (converter->ptype_view.vector_type == VECTOR_TYPE_LIST_OF) {
    set_converter_children_schema(converter_xptr, schema_xptr);
  }

  return NANOARROW_OK;
}

int nanoarrow_converter_set_array(SEXP converter_xptr, SEXP array_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowArray* array = array_from_xptr(array_xptr);
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewSetArray(&converter->array_view, array, &converter->error));
  SET_VECTOR_ELT(converter_shelter, 2, array_xptr);
  converter->src.offset = 0;
  converter->src.length = 0;

  if (converter->ptype_view.vector_type == VECTOR_TYPE_LIST_OF) {
    set_converter_children_array(converter_xptr, array_xptr);
  }

  return NANOARROW_OK;
}

int nanoarrow_converter_reserve(SEXP converter_xptr, R_xlen_t additional_size) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP current_result = VECTOR_ELT(converter_shelter, 4);

  if (current_result != R_NilValue) {
    ArrowErrorSet(&converter->error, "Reallocation in converter is not implemented");
    return ENOTSUP;
  }

  SEXP result_sexp;
  if (converter->ptype_view.ptype != R_NilValue) {
    result_sexp = PROTECT(
        nanoarrow_materialize_realloc(converter->ptype_view.ptype, additional_size));
  } else {
    result_sexp =
        PROTECT(nanoarrow_alloc_type(converter->ptype_view.vector_type, additional_size));
  }

  SET_VECTOR_ELT(converter_shelter, 4, result_sexp);
  UNPROTECT(1);

  converter->dst.vec_sexp = result_sexp;
  converter->dst.offset = 0;
  converter->dst.length = 0;
  converter->size = 0;
  converter->capacity = additional_size;

  return NANOARROW_OK;
}

R_xlen_t nanoarrow_converter_materialize_n(SEXP converter_xptr, R_xlen_t n) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  if ((converter->dst.offset + n) > converter->capacity) {
    n = converter->capacity - converter->dst.offset;
  }

  if ((converter->src.offset + n) > converter->array_view.array->length) {
    n = converter->array_view.array->length - converter->src.offset;
  }

  if (n == 0) {
    return 0;
  }

  converter->src.length = converter->dst.length = n;
  int result = nanoarrow_materialize(converter);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(&converter->error, "Error in nanoarrow_materialize()");
    return 0;
  }

  converter->src.offset += n;
  converter->dst.offset += n;
  converter->size += n;
  return n;
}

int nanoarrow_converter_materialize_all(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  R_xlen_t remaining = converter->array_view.array->length;
  NANOARROW_RETURN_NOT_OK(nanoarrow_converter_reserve(converter_xptr, remaining));
  if (nanoarrow_converter_materialize_n(converter_xptr, remaining) != remaining) {
    return ERANGE;
  } else {
    return NANOARROW_OK;
  }
}

int nanoarrow_converter_finalize(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP current_result = VECTOR_ELT(converter_shelter, 4);

  // Materialize never called (e.g., empty stream)
  if (current_result == R_NilValue) {
    NANOARROW_RETURN_NOT_OK(nanoarrow_converter_reserve(converter_xptr, 0));
    current_result = VECTOR_ELT(converter_shelter, 4);
  }

  // Check result size. A future implementation could also shrink the length
  // or reallocate a shorter vector.
  R_xlen_t current_result_size = nanoarrow_vec_size(current_result);
  if (current_result_size != converter->size) {
    ArrowErrorSet(&converter->error,
                  "Expected result of size %ld but got result of size %ld",
                  (long)current_result_size, (long)converter->size);
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

SEXP nanoarrow_converter_result(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP result = PROTECT(VECTOR_ELT(converter_shelter, 4));
  SET_VECTOR_ELT(converter_shelter, 4, R_NilValue);
  converter->dst.vec_sexp = R_NilValue;
  converter->dst.offset = 0;
  converter->dst.length = 0;
  converter->size = 0;
  converter->capacity = 0;
  UNPROTECT(1);
  return result;
}

void nanoarrow_converter_stop(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  Rf_error("%s", ArrowErrorMessage(&converter->error));
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
      SET_STRING_ELT(dst->vec_sexp, i, NA_STRING);
    } else {
      item = ArrowArrayViewGetStringUnsafe(src->array_view, src->offset + i);
      SET_STRING_ELT(dst->vec_sexp, i, Rf_mkCharLenCE(item.data, item.n_bytes, CE_UTF8));
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
    default:
      return ENOTSUP;
  }
}
