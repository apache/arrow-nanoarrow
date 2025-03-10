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

#include <limits.h>

#include "array.h"
#include "buffer.h"
#include "materialize.h"
#include "nanoarrow.h"
#include "schema.h"
#include "util.h"

static void call_as_nanoarrow_array(SEXP x_sexp, struct ArrowArray* array,
                                    SEXP schema_xptr, const char* fun_name) {
  SEXP fun = PROTECT(Rf_install(fun_name));
  SEXP call = PROTECT(Rf_lang3(fun, x_sexp, schema_xptr));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));

  // In many cases we can skip the array_export() step (which adds some complexity
  // and an additional R object to the mix)
  if (Rf_inherits(result, "nanoarrow_array_dont_export")) {
    struct ArrowArray* array_result = nanoarrow_array_from_xptr(result);
    ArrowArrayMove(array_result, array);
  } else {
    array_export(result, array);
  }

  UNPROTECT(3);
}

static SEXP call_storage_integer_for_decimal(SEXP x_sexp, int scale) {
  SEXP scale_sexp = PROTECT(Rf_ScalarInteger(scale));
  SEXP fun = PROTECT(Rf_install("storage_integer_for_decimal"));
  SEXP call = PROTECT(Rf_lang3(fun, x_sexp, scale_sexp));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));
  UNPROTECT(4);
  return result;
}

static void as_decimal_array(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowSchemaView* schema_view,
                             struct ArrowError* error);

static void as_array_int(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  // Consider integer -> numeric types that are easy to implement
  switch (schema_view->type) {
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      as_decimal_array(x_sexp, array, schema_xptr, schema_view, error);
      return;
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_INT32:
      break;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }

  // We don't consider altrep for now: we need an array of int32_t, and while we
  // *could* avoid materializing, there's no point because the source altrep
  // object almost certainly knows how to do this faster than we do.
  int* x_data = INTEGER(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  int result = ArrowArrayInitFromType(array, schema_view->type);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  if (schema_view->type == NANOARROW_TYPE_INT32) {
    // Zero-copy create: just borrow the data buffer
    buffer_borrowed(ArrowArrayBuffer(array, 1), x_data, len * sizeof(int32_t), x_sexp);
  } else {
    // Otherwise, use the integer appender
    result = ArrowArrayStartAppending(array);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayStartAppending() failed");
    }

    result = ArrowArrayReserve(array, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayReserve() failed");
    }

    for (int64_t i = 0; i < len; i++) {
      result = ArrowArrayAppendInt(array, x_data[i]);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendInt() failed");
      }
    }
  }

  // Set the array fields
  array->length = len;
  array->offset = 0;
  int64_t null_count = 0;

  // Look for the first null (will be the last index if there are none)
  int64_t first_null = -1;
  for (int64_t i = 0; i < len; i++) {
    if (x_data[i] == NA_INTEGER) {
      first_null = i;
      break;
    }
  }

  // If there are nulls, pack the validity buffer
  if (first_null != -1) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    ArrowBitmapAppendUnsafe(&bitmap, 1, first_null);
    for (int64_t i = first_null; i < len; i++) {
      uint8_t is_valid = x_data[i] != NA_INTEGER;
      null_count += !is_valid;
      ArrowBitmapAppendUnsafe(&bitmap, is_valid, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }

  array->null_count = null_count;

  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault(): %s", error->message);
  }
}

static void as_array_lgl(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  // We can zero-copy convert to int32
  if (schema_view->type == NANOARROW_TYPE_INT32) {
    as_array_int(x_sexp, array, schema_xptr, schema_view, error);
    return;
  }

  // Only consider bool for now
  if (schema_view->type != NANOARROW_TYPE_BOOL) {
    call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
    return;
  }

  int* x_data = INTEGER(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  int result = ArrowArrayInitFromType(array, NANOARROW_TYPE_BOOL);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  struct ArrowBitmap value_bitmap;
  ArrowBitmapInit(&value_bitmap);
  result = ArrowBitmapReserve(&value_bitmap, len);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowBitmapReserve() failed");
  }

  int has_nulls = 0;
  for (int64_t i = 0; i < len; i++) {
    if (x_data[i] == NA_INTEGER) {
      has_nulls = 1;
      ArrowBitmapAppendUnsafe(&value_bitmap, 0, 1);
    } else {
      ArrowBitmapAppendUnsafe(&value_bitmap, x_data[i] != 0, 1);
    }
  }

  result = ArrowArraySetBuffer(array, 1, &value_bitmap.buffer);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArraySetBuffer() failed");
  }

  // Set the array fields
  array->length = len;
  array->offset = 0;
  int64_t null_count = 0;

  // If there are nulls, pack the validity buffer
  if (has_nulls) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    for (int64_t i = 0; i < len; i++) {
      uint8_t is_valid = x_data[i] != NA_INTEGER;
      null_count += !is_valid;
      ArrowBitmapAppendUnsafe(&bitmap, is_valid, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }

  array->null_count = null_count;
  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault(): %s", error->message);
  }
}

static void as_array_dbl(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  // Consider double -> na_double() and double -> na_int64()/na_int32()
  // (mostly so that we can support date/time types with various units)
  switch (schema_view->type) {
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      as_decimal_array(x_sexp, array, schema_xptr, schema_view, error);
      return;
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_INT32:
      break;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }

  double* x_data = REAL(x_sexp);
  int64_t len = Rf_xlength(x_sexp);

  int result = ArrowArrayInitFromType(array, schema_view->type);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  if (schema_view->type == NANOARROW_TYPE_DOUBLE) {
    // Just borrow the data buffer (zero-copy)
    buffer_borrowed(ArrowArrayBuffer(array, 1), x_data, len * sizeof(double), x_sexp);

  } else if (schema_view->type == NANOARROW_TYPE_INT64) {
    // double -> int64_t
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, 1);
    result = ArrowBufferReserve(buffer, len * sizeof(int64_t));
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBufferReserve() failed");
    }

    int64_t* buffer_data = (int64_t*)buffer->data;
    for (int64_t i = 0; i < len; i++) {
      // UBSAN warns for buffer_data[i] = nan
      if (R_IsNA(x_data[i]) || R_IsNaN(x_data[i])) {
        buffer_data[i] = 0;
      } else {
        buffer_data[i] = (int64_t)x_data[i];
      }
    }

    buffer->size_bytes = len * sizeof(int64_t);

  } else if (schema_view->type == NANOARROW_TYPE_INT32) {
    // double -> int32_t
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, 1);
    result = ArrowBufferReserve(buffer, len * sizeof(int32_t));
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBufferReserve() failed");
    }

    int32_t* buffer_data = (int32_t*)buffer->data;

    // It's easy to accidentally overflow here, so make sure to warn
    int64_t n_overflow = 0;
    for (int64_t i = 0; i < len; i++) {
      // UBSAN warns for buffer_data[i] = nan
      if (R_IsNA(x_data[i]) || R_IsNaN(x_data[i])) {
        buffer_data[i] = 0;
      } else if (x_data[i] > INT_MAX || x_data[i] < INT_MIN) {
        n_overflow++;
        buffer_data[i] = 0;
      } else {
        buffer_data[i] = (int32_t)x_data[i];
      }
    }

    if (n_overflow > 0) {
      warn_lossy_conversion(n_overflow, "overflowed in double -> na_int32() creation");
    }

    buffer->size_bytes = len * sizeof(int32_t);
  } else {
    result = ArrowArrayStartAppending(array);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayStartAppending() failed");
    }

    result = ArrowArrayReserve(array, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayReserve() failed");
    }

    for (int64_t i = 0; i < len; i++) {
      result = ArrowArrayAppendDouble(array, x_data[i]);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendDouble() failed");
      }
    }
  }

  // Set the array fields
  array->length = len;
  array->offset = 0;
  int64_t null_count = 0;

  // Look for the first null (will be the last index if there are none)
  int64_t first_null = -1;
  for (int64_t i = 0; i < len; i++) {
    if (R_IsNA(x_data[i]) || R_IsNaN(x_data[i])) {
      first_null = i;
      break;
    }
  }

  // If there are nulls, pack the validity buffer
  if (first_null != -1) {
    struct ArrowBitmap bitmap;
    ArrowBitmapInit(&bitmap);
    result = ArrowBitmapReserve(&bitmap, len);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowBitmapReserve() failed");
    }

    ArrowBitmapAppendUnsafe(&bitmap, 1, first_null);
    for (int64_t i = first_null; i < len; i++) {
      uint8_t is_valid = !R_IsNA(x_data[i]) && !R_IsNaN(x_data[i]);
      null_count += !is_valid;
      ArrowBitmapAppendUnsafe(&bitmap, is_valid, 1);
    }

    ArrowArraySetValidityBitmap(array, &bitmap);
  }

  array->null_count = null_count;
  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault(): %s", error->message);
  }
}

static void as_decimal_array(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowSchemaView* schema_view,
                             struct ArrowError* error) {
  // Use R to generate the input we need for ArrowDecimalSetDigits()
  SEXP x_digits_sexp =
      PROTECT(call_storage_integer_for_decimal(x_sexp, schema_view->decimal_scale));

  struct ArrowDecimal item;
  ArrowDecimalInit(&item, schema_view->decimal_bitwidth, schema_view->decimal_precision,
                   schema_view->decimal_scale);

  int result = ArrowArrayInitFromType(array, schema_view->type);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  result = ArrowArrayStartAppending(array);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStartAppending() failed");
  }

  int64_t len = Rf_xlength(x_sexp);
  result = ArrowArrayReserve(array, len);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayReserve() failed");
  }

  struct ArrowStringView item_digits_view;
  for (int64_t i = 0; i < len; i++) {
    SEXP item_sexp = STRING_ELT(x_digits_sexp, i);
    if (item_sexp == NA_STRING) {
      result = ArrowArrayAppendNull(array, 1);
    } else {
      item_digits_view.data = CHAR(item_sexp);
      item_digits_view.size_bytes = Rf_length(item_sexp);
      ArrowDecimalSetDigits(&item, item_digits_view);
      result = ArrowArrayAppendDecimal(array, &item);
    }

    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayAppendDecimal() failed");
    }
  }

  UNPROTECT(1);

  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault(): %s", error->message);
  }
}

static void as_array_chr(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                         struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  switch (schema_view->type) {
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_STRING_VIEW:
    case NANOARROW_TYPE_BINARY_VIEW:
      break;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }

  int64_t len = Rf_xlength(x_sexp);

  int result = ArrowArrayInitFromType(array, schema_view->type);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  result = ArrowArrayStartAppending(array);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStartAppending() failed");
  }

  struct ArrowStringView item_view;
  for (int64_t i = 0; i < len; i++) {
    SEXP item = STRING_ELT(x_sexp, i);

    if (item == NA_STRING) {
      result = ArrowArrayAppendNull(array, 1);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendString() failed");
      }
    } else {
      const void* vmax = vmaxget();
      item_view.data = Rf_translateCharUTF8(item);
      item_view.size_bytes = strlen(item_view.data);
      result = ArrowArrayAppendString(array, item_view);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendString() failed");
      }

      vmaxset(vmax);
    }
  }

  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault() failed with code %d: %s", result,
             error->message);
  }
}

static void as_array_default(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowError* error);

static void as_array_data_frame(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                                struct ArrowSchemaView* schema_view,
                                struct ArrowError* error) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

  switch (schema_view->type) {
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "union_array_from_data_frame");
      return;
    case NANOARROW_TYPE_STRUCT:
      break;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }

  if (Rf_xlength(x_sexp) != schema->n_children) {
    Rf_error("Expected %ld schema children but found %ld", (long)Rf_xlength(x_sexp),
             (long)schema->n_children);
  }

  int result = ArrowArrayInitFromType(array, NANOARROW_TYPE_STRUCT);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed");
  }

  result = ArrowArrayAllocateChildren(array, schema->n_children);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayAllocateChildren() failed");
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    SEXP child_xptr = PROTECT(borrow_schema_child_xptr(schema_xptr, i));
    as_array_default(VECTOR_ELT(x_sexp, i), array->children[i], child_xptr, error);
    UNPROTECT(1);
  }

  array->length = nanoarrow_data_frame_size(x_sexp);
  array->null_count = 0;
  array->offset = 0;
}

static void as_array_list(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                          struct ArrowSchemaView* schema_view, struct ArrowError* error) {
  switch (schema_view->type) {
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_BINARY_VIEW:
      break;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }

  int64_t len = Rf_xlength(x_sexp);

  // Use schema here to ensure we fixed-size binary byte width works
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);
  int result = ArrowArrayInitFromSchema(array, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromType() failed: %s", error->message);
  }

  result = ArrowArrayStartAppending(array);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayStartAppending() failed");
  }

  struct ArrowBufferView item_view;
  for (int64_t i = 0; i < len; i++) {
    SEXP item = VECTOR_ELT(x_sexp, i);

    if (item == R_NilValue) {
      result = ArrowArrayAppendNull(array, 1);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendNull() failed");
      }
    } else if (TYPEOF(item) == RAWSXP) {
      item_view.data.data = RAW(item);
      item_view.size_bytes = Rf_xlength(item);
      result = ArrowArrayAppendBytes(array, item_view);
      if (result != NANOARROW_OK) {
        Rf_error("ArrowArrayAppendBytes() failed");
      }
    } else {
      Rf_error("All list items must be raw() or NULL in conversion to %s",
               ArrowTypeString(schema_view->type));
    }
  }

  result = ArrowArrayFinishBuildingDefault(array, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault() failed with code %d: %s", result,
             error->message);
  }
}

static void as_array_default(SEXP x_sexp, struct ArrowArray* array, SEXP schema_xptr,
                             struct ArrowError* error) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowSchemaViewInit(): %s", error->message);
  }

  // Ensure that extension types dispatch from R regardless of source
  if (schema_view.extension_name.size_bytes > 0) {
    call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
    return;
  }

  if (Rf_isObject(x_sexp)) {
    if (Rf_inherits(x_sexp, "data.frame")) {
      as_array_data_frame(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    } else {
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
    }
  }

  switch (TYPEOF(x_sexp)) {
    case LGLSXP:
      as_array_lgl(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    case INTSXP:
      as_array_int(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    case REALSXP:
      as_array_dbl(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    case STRSXP:
      as_array_chr(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    case VECSXP:
      as_array_list(x_sexp, array, schema_xptr, &schema_view, error);
      return;
    default:
      call_as_nanoarrow_array(x_sexp, array, schema_xptr, "as_nanoarrow_array_from_c");
      return;
  }
}

SEXP nanoarrow_c_as_array_default(SEXP x_sexp, SEXP schema_xptr) {
  SEXP array_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* array = nanoarrow_output_array_from_xptr(array_xptr);
  struct ArrowError error;

  as_array_default(x_sexp, array, schema_xptr, &error);
  array_xptr_set_schema(array_xptr, schema_xptr);
  UNPROTECT(1);
  return array_xptr;
}
