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
#include "nanoarrow.h"
#include "schema.h"
#include "util.h"

SEXP nanoarrow_c_array_init(SEXP schema_xptr) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

  SEXP array_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* array = nanoarrow_output_array_from_xptr(array_xptr);

  struct ArrowError error;
  int result = ArrowArrayInitFromSchema(array, schema, &error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromSchema(): %s", error.message);
  }

  array_xptr_set_schema(array_xptr, schema_xptr);
  UNPROTECT(1);
  return array_xptr;
}

SEXP nanoarrow_c_array_set_length(SEXP array_xptr, SEXP length_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  if (TYPEOF(length_sexp) != REALSXP || Rf_length(length_sexp) != 1) {
    Rf_error("array$length must be double(1)");
  }

  double length = REAL(length_sexp)[0];
  if (ISNA(length) || ISNAN(length) || length < 0) {
    Rf_error("array$length must be finite and greater than zero");
  }

  array->length = (int64_t)length;
  return R_NilValue;
}

SEXP nanoarrow_c_array_set_null_count(SEXP array_xptr, SEXP null_count_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  if (TYPEOF(null_count_sexp) != REALSXP || Rf_length(null_count_sexp) != 1) {
    Rf_error("array$null_count must be double(1)");
  }

  double null_count = REAL(null_count_sexp)[0];
  if (ISNA(null_count) || ISNAN(null_count) || null_count < -1) {
    Rf_error("array$null_count must be finite and greater than -1");
  }

  array->null_count = (int64_t)null_count;
  return R_NilValue;
}

SEXP nanoarrow_c_array_set_offset(SEXP array_xptr, SEXP offset_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  if (TYPEOF(offset_sexp) != REALSXP || Rf_length(offset_sexp) != 1) {
    Rf_error("array$offset must be double(1)");
  }

  double offset = REAL(offset_sexp)[0];
  if (ISNA(offset) || ISNAN(offset) || offset < 0) {
    Rf_error("array$offset must be finite and greater than zero");
  }

  array->offset = (int64_t)offset;
  return R_NilValue;
}

SEXP nanoarrow_c_array_set_buffers(SEXP array_xptr, SEXP buffers_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);

  int64_t n_buffers = Rf_xlength(buffers_sexp);
  if (n_buffers != array->n_buffers) {
    Rf_error("Changing the number of buffers in array_modify is not supported");
  }

  // Release any buffers that aren't about to be replaced
  for (int64_t i = n_buffers; i < array->n_buffers; i++) {
    ArrowBufferReset(ArrowArrayBuffer(array, i));
  }

  array->n_buffers = n_buffers;
  for (int64_t i = 0; i < n_buffers; i++) {
    SEXP buffer_xptr = VECTOR_ELT(buffers_sexp, i);
    struct ArrowBuffer* src = buffer_from_xptr(buffer_xptr);

    // We can't necessarily ArrowBufferMove(src) because that buffer might
    // have been pointed at by something else. So, we do this slightly awkward
    // dance to make sure buffer_xptr stays valid after this call.
    SEXP buffer_xptr_clone =
        PROTECT(buffer_borrowed_xptr(src->data, src->size_bytes, buffer_xptr));
    struct ArrowBuffer* src_clone =
        (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr_clone);

    // Release whatever buffer is currently there and replace it with src_clone
    ArrowBufferReset(ArrowArrayBuffer(array, i));
    int result = ArrowArraySetBuffer(array, i, src_clone);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArraySetBuffer() failed");
    }

    UNPROTECT(1);
  }

  return R_NilValue;
}

static void release_all_children(struct ArrowArray* array) {
  for (int64_t i = 0; i < array->n_children; i++) {
    if (array->children[i]->release != NULL) {
      array->children[i]->release(array->children[i]);
    }
  }
}

static void free_all_children(struct ArrowArray* array) {
  for (int64_t i = 0; i < array->n_children; i++) {
    if (array->children[i] != NULL) {
      ArrowFree(array->children[i]);
      array->children[i] = NULL;
    }
  }

  if (array->children != NULL) {
    ArrowFree(array->children);
    array->children = NULL;
  }

  array->n_children = 0;
}

SEXP nanoarrow_c_array_set_children(SEXP array_xptr, SEXP children_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);

  release_all_children(array);

  if (Rf_xlength(children_sexp) == 0) {
    free_all_children(array);
    return R_NilValue;
  }

  if (Rf_xlength(children_sexp) != array->n_children) {
    free_all_children(array);
    int result = ArrowArrayAllocateChildren(array, Rf_xlength(children_sexp));
    if (result != NANOARROW_OK) {
      Rf_error("Error allocating array$children of size %ld",
               (long)Rf_xlength(children_sexp));
    }
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    // The arrays here will be moved, invalidating the arrays in the passed
    // list (the export step is handled in R)
    SEXP child_xptr = VECTOR_ELT(children_sexp, i);
    struct ArrowArray* child = nanoarrow_array_from_xptr(child_xptr);
    ArrowArrayMove(child, array->children[i]);
  }

  return R_NilValue;
}

SEXP nanoarrow_c_array_set_dictionary(SEXP array_xptr, SEXP dictionary_xptr) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);

  // If there's already a dictionary, make sure we release it
  if (array->dictionary != NULL) {
    if (array->dictionary->release != NULL) {
      array->dictionary->release(array->dictionary);
    }
  }

  if (dictionary_xptr == R_NilValue) {
    if (array->dictionary != NULL) {
      ArrowFree(array->dictionary);
      array->dictionary = NULL;
    }
  } else {
    if (array->dictionary == NULL) {
      int result = ArrowArrayAllocateDictionary(array);
      if (result != NANOARROW_OK) {
        Rf_error("Error allocating array$dictionary");
      }
    }

    struct ArrowArray* dictionary = nanoarrow_array_from_xptr(dictionary_xptr);
    ArrowArrayMove(dictionary, array->dictionary);
  }

  return R_NilValue;
}

static int move_array_buffers(struct ArrowArray* src, struct ArrowArray* dst,
                              struct ArrowSchema* schema, struct ArrowError* error) {
  error->message[0] = '\0';
  dst->length = src->length;
  dst->null_count = src->null_count;
  dst->offset = src->offset;

  if (src->n_buffers != dst->n_buffers) {
    ArrowErrorSet(error, "Expected %ld buffer(s) but got %ld", (long)dst->n_buffers,
                  (long)src->n_buffers);
    return EINVAL;
  }

  for (int64_t i = 0; i < src->n_buffers; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArraySetBuffer(dst, i, ArrowArrayBuffer(src, i)));
  }

  if (src->n_children != dst->n_children) {
    ArrowErrorSet(error, "Expected %ld child(ren) but got %ld", (long)dst->n_children,
                  (long)src->n_children);
    return EINVAL;
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(move_array_buffers(src->children[i], dst->children[i],
                                               schema->children[i], error));
  }

  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(
        move_array_buffers(src->dictionary, dst->dictionary, schema->dictionary, error));
  }

  return NANOARROW_OK;
}

SEXP nanoarrow_c_array_validate_after_modify(SEXP array_xptr, SEXP schema_xptr) {
  // A very particular type of validation we can do with the ArrowArray we use
  // in nanoarrow_array_modify() (which was created using ArrowArrayInit).
  // At this point we know how long each buffer is (via ArrowArrayBuffer())
  // but after we send the array into the wild, that information is lost.
  // This operation will invalidate array_xptr (but this is OK since we very
  // specifically just allocated it).
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);
  struct ArrowError error;

  // Even though array was initialized using ArrowArrayInit(), it doesn't have
  // all the information about storage types since it didn't necessarily know
  // what the storage type would be when it was being constructed. Here we create
  // a version that does and move buffers recursively into it.
  SEXP array_dst_xptr = PROTECT(nanoarrow_array_owning_xptr());
  struct ArrowArray* array_dst = nanoarrow_output_array_from_xptr(array_dst_xptr);

  int result = ArrowArrayInitFromSchema(array_dst, schema, &error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayInitFromSchema(): %s", error.message);
  }

  // Add any variadic buffers that might be required
  if ((array->n_buffers > array_dst->n_buffers) &&
      array->n_buffers > NANOARROW_MAX_FIXED_BUFFERS) {
    result =
        ArrowArrayAddVariadicBuffers(array_dst, array->n_buffers - array_dst->n_buffers);
    if (result != NANOARROW_OK) {
      Rf_error("ArrowArrayAddVariadicBuffers() failed");
    }
  }

  result = move_array_buffers(array, array_dst, schema, &error);
  if (result != NANOARROW_OK) {
    Rf_error("move_array_buffers: %s", error.message);
  }

  result = ArrowArrayFinishBuildingDefault(array_dst, &error);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowArrayFinishBuildingDefault(): %s", error.message);
  }

  UNPROTECT(1);
  return array_dst_xptr;
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
    struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
    struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);

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
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
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

static SEXP borrow_array_view_dictionary(struct ArrowArrayView* array_view,
                                         SEXP shelter) {
  if (array_view != NULL) {
    return R_MakeExternalPtr(array_view->dictionary, R_NilValue, shelter);
  } else {
    return R_NilValue;
  }
}

static SEXP borrow_unknown_buffer(struct ArrowArray* array, int64_t i, SEXP shelter) {
  return buffer_borrowed_xptr(array->buffers[i], 0, shelter);
}

static SEXP borrow_buffer(struct ArrowArrayView* array_view, int64_t i, SEXP shelter) {
  SEXP buffer_class = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(buffer_class, 1, Rf_mkChar("nanoarrow_buffer"));

  struct ArrowBufferView view = ArrowArrayViewGetBufferView(array_view, i);
  enum ArrowBufferType buffer_type = ArrowArrayViewGetBufferType(array_view, i);
  enum ArrowType data_type = ArrowArrayViewGetBufferDataType(array_view, i);
  int64_t element_size_bits = ArrowArrayViewGetBufferElementSizeBits(array_view, i);

  SEXP buffer_xptr =
      PROTECT(buffer_borrowed_xptr(view.data.data, view.size_bytes, shelter));

  buffer_borrowed_xptr_set_type(buffer_xptr, buffer_type, data_type, element_size_bits);
  UNPROTECT(2);
  return buffer_xptr;
}

SEXP nanoarrow_c_array_proxy(SEXP array_xptr, SEXP array_view_xptr, SEXP recursive_sexp) {
  struct ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  int recursive = LOGICAL(recursive_sexp)[0];
  struct ArrowArrayView* array_view = NULL;
  if (array_view_xptr != R_NilValue) {
    array_view = (struct ArrowArrayView*)R_ExternalPtrAddr(array_view_xptr);
  }

  const char* names[] = {"length",   "null_count", "offset", "buffers",
                         "children", "dictionary", ""};
  SEXP array_proxy = PROTECT(Rf_mkNamed(VECSXP, names));

  SET_VECTOR_ELT(array_proxy, 0, length_sexp_from_int64(array->length));
  SET_VECTOR_ELT(array_proxy, 1, length_sexp_from_int64(array->null_count));
  SET_VECTOR_ELT(array_proxy, 2, length_sexp_from_int64(array->offset));

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

  if (array->dictionary != NULL) {
    SEXP dictionary_xptr = PROTECT(borrow_array_xptr(array->dictionary, array_xptr));

    if (recursive) {
      SEXP dictionary_view_xptr =
          PROTECT(borrow_array_view_dictionary(array_view, array_view_xptr));
      SEXP dictionary_proxy = PROTECT(
          nanoarrow_c_array_proxy(dictionary_xptr, dictionary_view_xptr, recursive_sexp));
      SET_VECTOR_ELT(array_proxy, 5, dictionary_proxy);
      UNPROTECT(2);
    } else {
      SET_VECTOR_ELT(array_proxy, 5, dictionary_xptr);
    }

    UNPROTECT(1);
  }

  UNPROTECT(1);
  return array_proxy;
}
