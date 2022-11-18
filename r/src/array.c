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

#include "array.h"
#include "nanoarrow.h"
#include "schema.h"
#include "util.h"

void finalize_array_xptr(SEXP array_xptr) {
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array != NULL && array->release != NULL) {
    array->release(array);
  }

  if (array != NULL) {
    ArrowFree(array);
  }
}

SEXP nanoarrow_c_array_set_schema(SEXP array_xptr, SEXP schema_xptr, SEXP validate_sexp) {
  // Fair game to remove a schema from a pointer
  if (schema_xptr == R_NilValue) {
    array_xptr_set_schema(array_xptr, R_NilValue);
    return R_NilValue;
  }

  int validate = LOGICAL(validate_sexp)[0];
  if (validate) {
    // If adding a schema, validate the schema and the pair
    struct ArrowArray* array = array_from_xptr(array_xptr);
    struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

    struct ArrowArrayView array_view;
    struct ArrowError error;
    int result = ArrowArrayViewInitFromSchema(&array_view, schema, &error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(&array_view);
      Rf_error("%s", ArrowErrorMessage(&error));
    }

    result = ArrowArrayViewSetArray(&array_view, array, &error);
    ArrowArrayViewReset(&array_view);
    if (result != NANOARROW_OK) {
      Rf_error("%s", ArrowErrorMessage(&error));
    }
  }

  array_xptr_set_schema(array_xptr, schema_xptr);
  return R_NilValue;
}

SEXP nanoarrow_c_infer_schema_array(SEXP array_xptr) {
  SEXP maybe_schema_xptr = R_ExternalPtrTag(array_xptr);
  if (Rf_inherits(maybe_schema_xptr, "nanoarrow_schema")) {
    return maybe_schema_xptr;
  } else {
    return R_NilValue;
  }
}

static SEXP borrow_array_xptr(struct ArrowArray* array, SEXP shelter) {
  SEXP array_xptr = PROTECT(R_MakeExternalPtr(array, R_NilValue, shelter));
  Rf_setAttrib(array_xptr, R_ClassSymbol, nanoarrow_cls_array);
  UNPROTECT(1);
  return array_xptr;
}

SEXP borrow_array_child_xptr(SEXP array_xptr, int64_t i) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  SEXP schema_xptr = R_ExternalPtrTag(array_xptr);
  SEXP child_xptr = PROTECT(borrow_array_xptr(array->children[i], array_xptr));
  if (schema_xptr != R_NilValue) {
    array_xptr_set_schema(child_xptr, borrow_schema_child_xptr(schema_xptr, i));
  }
  UNPROTECT(1);
  return child_xptr;
}

static SEXP borrow_array_view_child(struct ArrowArrayView* array_view, int64_t i,
                                    SEXP shelter) {
  if (array_view != NULL) {
    return R_MakeExternalPtr(array_view->children[i], R_NilValue, shelter);
  } else {
    return R_NilValue;
  }
}

static SEXP borrow_unknown_buffer(struct ArrowArray* array, int64_t i, SEXP shelter) {
  SEXP buffer_class = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(buffer_class, 0, Rf_mkChar("nanoarrow_buffer_unknown"));
  SET_STRING_ELT(buffer_class, 1, Rf_mkChar("nanoarrow_buffer"));

  SEXP buffer = PROTECT(R_MakeExternalPtr((void*)array->buffers[i], R_NilValue, shelter));
  Rf_setAttrib(buffer, R_ClassSymbol, buffer_class);
  UNPROTECT(2);
  return buffer;
}

static SEXP length_from_int64(int64_t value) {
  if (value < 2147483647) {
    return Rf_ScalarInteger(value);
  } else {
    return Rf_ScalarReal(value);
  }
}

static SEXP borrow_buffer(struct ArrowArrayView* array_view, int64_t i, SEXP shelter) {
  SEXP buffer_class = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(buffer_class, 1, Rf_mkChar("nanoarrow_buffer"));

  const char* class0 = "nanoarrow_buffer_unknown";

  switch (array_view->layout.buffer_type[i]) {
    case NANOARROW_BUFFER_TYPE_VALIDITY:
      class0 = "nanoarrow_buffer_validity";
      break;
    case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
      switch (array_view->layout.element_size_bits[i]) {
        case 32:
          class0 = "nanoarrow_buffer_data_offset32";
          break;
        case 64:
          class0 = "nanoarrow_buffer_data_offset64";
          break;
        default:
          break;
      }
      break;
    case NANOARROW_BUFFER_TYPE_DATA:
      switch (array_view->storage_type) {
        case NANOARROW_TYPE_BOOL:
          class0 = "nanoarrow_buffer_data_bool";
          break;
        case NANOARROW_TYPE_UINT8:
          class0 = "nanoarrow_buffer_data_uint8";
          break;
        case NANOARROW_TYPE_INT8:
          class0 = "nanoarrow_buffer_data_int8";
          break;
        case NANOARROW_TYPE_UINT16:
          class0 = "nanoarrow_buffer_data_uint16";
          break;
        case NANOARROW_TYPE_INT16:
          class0 = "nanoarrow_buffer_data_int16";
          break;
        case NANOARROW_TYPE_UINT32:
          class0 = "nanoarrow_buffer_data_uint32";
          break;
        case NANOARROW_TYPE_INT32:
          class0 = "nanoarrow_buffer_data_int32";
          break;
        case NANOARROW_TYPE_UINT64:
          class0 = "nanoarrow_buffer_data_uint64";
          break;
        case NANOARROW_TYPE_INT64:
          class0 = "nanoarrow_buffer_data_int64";
          break;
        case NANOARROW_TYPE_HALF_FLOAT:
          class0 = "nanoarrow_buffer_data_half_float";
          break;
        case NANOARROW_TYPE_FLOAT:
          class0 = "nanoarrow_buffer_data_float";
          break;
        case NANOARROW_TYPE_DOUBLE:
          class0 = "nanoarrow_buffer_data_double";
          break;
        case NANOARROW_TYPE_DECIMAL128:
          class0 = "nanoarrow_buffer_data_decimal128";
          break;
        case NANOARROW_TYPE_DECIMAL256:
          class0 = "nanoarrow_buffer_data_decimal256";
          break;
        case NANOARROW_TYPE_INTERVAL_MONTHS:
          class0 = "nanoarrow_buffer_data_int32";
          break;
        case NANOARROW_TYPE_INTERVAL_DAY_TIME:
          class0 = "nanoarrow_buffer_data_int64";
          break;
        case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
          class0 = "nanoarrow_buffer_data_interval_month_day_nano";
          break;
        case NANOARROW_TYPE_STRING:
        case NANOARROW_TYPE_LARGE_STRING:
          class0 = "nanoarrow_buffer_data_utf8";
          break;
        case NANOARROW_TYPE_FIXED_SIZE_BINARY:
        case NANOARROW_TYPE_BINARY:
        case NANOARROW_TYPE_LARGE_BINARY:
          class0 = "nanoarrow_buffer_data_uint8";
          break;
        default:
          break;
      }
      break;
    case NANOARROW_BUFFER_TYPE_TYPE_ID:
      class0 = "nanoarrow_buffer_type_id";
      break;
    case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
      class0 = "nanoarrow_buffer_union_offset";
      break;
    default:
      break;
  }

  SET_STRING_ELT(buffer_class, 0, Rf_mkChar(class0));

  const char* names[] = {"size_bytes", "element_size_bits", ""};
  SEXP buffer_info = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(buffer_info, 0, length_from_int64(array_view->buffer_views[i].n_bytes));
  SET_VECTOR_ELT(buffer_info, 1,
                 length_from_int64(array_view->layout.element_size_bits[i]));

  SEXP buffer = PROTECT(R_MakeExternalPtr((void*)array_view->buffer_views[i].data.data,
                                          buffer_info, shelter));
  Rf_setAttrib(buffer, R_ClassSymbol, buffer_class);
  UNPROTECT(3);
  return buffer;
}

SEXP nanoarrow_c_array_proxy(SEXP array_xptr, SEXP array_view_xptr, SEXP recursive_sexp) {
  struct ArrowArray* array = array_from_xptr(array_xptr);
  int recursive = LOGICAL(recursive_sexp)[0];
  struct ArrowArrayView* array_view = NULL;
  if (array_view_xptr != R_NilValue) {
    array_view = (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  }

  const char* names[] = {"length",   "null_count", "offset", "buffers",
                         "children", "dictionary", ""};
  SEXP array_proxy = PROTECT(Rf_mkNamed(VECSXP, names));

  SET_VECTOR_ELT(array_proxy, 0, length_from_int64(array->length));
  SET_VECTOR_ELT(array_proxy, 1, length_from_int64(array->null_count));
  SET_VECTOR_ELT(array_proxy, 2, length_from_int64(array->offset));

  if (array->n_buffers > 0) {
    SEXP buffers = PROTECT(Rf_allocVector(VECSXP, array->n_buffers));
    for (int64_t i = 0; i < array->n_buffers; i++) {
      if (array_view != NULL) {
        SET_VECTOR_ELT(buffers, i, borrow_buffer(array_view, i, array_xptr));
      } else {
        SET_VECTOR_ELT(buffers, i, borrow_unknown_buffer(array, i, array_xptr));
      }
    }

    SET_VECTOR_ELT(array_proxy, 3, buffers);
    UNPROTECT(1);
  }

  if (array->n_children > 0) {
    SEXP children = PROTECT(Rf_allocVector(VECSXP, array->n_children));
    for (int64_t i = 0; i < array->n_children; i++) {
      SEXP child = PROTECT(borrow_array_xptr(array->children[i], array_xptr));
      if (recursive) {
        SEXP array_view_child =
            PROTECT(borrow_array_view_child(array_view, i, array_view_xptr));
        SET_VECTOR_ELT(children, i,
                       nanoarrow_c_array_proxy(child, array_view_child, recursive_sexp));
        UNPROTECT(1);
      } else {
        SET_VECTOR_ELT(children, i, child);
      }
      UNPROTECT(1);
    }

    SET_VECTOR_ELT(array_proxy, 4, children);
    UNPROTECT(1);
  }

  // The recursive-ness of the dictionary is handled in R because this is not part
  // of the struct ArrowArrayView.
  if (array->dictionary != NULL) {
    SET_VECTOR_ELT(array_proxy, 5, borrow_array_xptr(array->dictionary, array_xptr));
  }

  UNPROTECT(1);
  return array_proxy;
}

// for ArrowArray* that are exported references to an R array_xptr
void finalize_exported_array(struct ArrowArray* array) {
  SEXP array_xptr = (SEXP)array->private_data;
  R_ReleaseObject(array_xptr);
  array->release = NULL;
}
