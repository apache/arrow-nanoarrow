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

#ifndef NANOARROW_R_H_INCLUDED
#define NANOARROW_R_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup nanoarrow-r Utilities for Arrow R extensions
///
/// EXPERIMENTAL: The interface and lifecycle semantics described in this header
/// should be considered experimental and may change in a future version based on
/// user feedback.
///
/// In the nanoarrow R package, an external pointer to an ArrowSchema, ArrowArray, or
/// ArrowArrayStream carries the class "nanoarrow_schema", "nanoarrow_array", or
/// "nanoarrow_array_stream" (respectively). The pointer must point to valid memory
/// or be NULL until the R external pointer object is finalized.
///
/// nanoarrow_schema_owning_xptr(), nanoarrow_array_owning_xptr(), and
/// nanoarrow_array_stream_owning_xptr() initialize such an external pointer using
/// malloc() and a NULL initial release() callback such that it can be distinguished from
/// a pointer to an initialized value according to the Arrow C Data/Stream interface
/// documentation. This structure is intended to have a valid value initialized into it
/// using ArrowXXXMove() or by passing the pointer to a suitable exporting function.
///
/// External pointers allocated by nanoarrow_xxxx_owning_xptr() register a finalizer
/// that will call the release() callback when its value is non-NULL and points to
/// a structure whose release() callback is also non-NULL. External pointers may also
/// manage lifecycle by declaring a strong reference to a single R object via
/// R_SetExternalPtrProtected(); however, when passing the address of an R external
/// pointer to a non-R library, the ownership of the structure must *not* have such SEXP
/// dependencies. The nanoarrow R package can wrap such an SEXP dependency into a
/// self-contained thread-safe release callback via nanoarrow_pointer_export() that
/// manages the SEXP dependency using a preserve/release mechanism similar to
/// R_PreserveObject()/ R_ReleaseObject().
///
/// The "tag" of an external pointer to an ArrowArray must be R_NilValue or an external
/// pointer to an ArrowSchema that may be used to interpret the pointed-to ArrowArray. The
/// "tag" of a nanoarrow external pointer to an ArrowSchema or ArrowArrayStream is
/// reserved for future use and must be R_NilValue.
///
/// @{

// Extra guard for versions of Arrow without the canonical guard
#ifndef ARROW_FLAG_DICTIONARY_ORDERED

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#include <stdint.h>

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callback to get the stream type
  // (will be the same for all arrays in the stream).
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowSchema must be released independently from the stream.
  int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);

  // Callback to get the next array
  // (if no error and the array is released, the stream has ended)
  //
  // Return value: 0 if successful, an `errno`-compatible error code otherwise.
  //
  // If successful, the ArrowArray must be released independently from the stream.
  int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);

  // Callback to get optional detailed error information.
  // This must only be called if the last stream operation failed
  // with a non-0 return code.
  //
  // Return value: pointer to a null-terminated character array describing
  // the last error, or NULL if no description is available.
  //
  // The returned pointer is only valid until the next operation on this stream
  // (including release).
  const char* (*get_last_error)(struct ArrowArrayStream*);

  // Release callback: release the stream's own resources.
  // Note that arrays returned by `get_next` must be individually released.
  void (*release)(struct ArrowArrayStream*);

  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_STREAM_INTERFACE
#endif  // ARROW_FLAG_DICTIONARY_ORDERED

/// \brief Allocate an external pointer to an ArrowSchema
///
/// Allocate an external pointer to an uninitialized ArrowSchema with a finalizer that
/// ensures that any non-null release callback in a pointed-to structure will be called
/// when the external pointer is garbage collected.
static inline SEXP nanoarrow_schema_owning_xptr(void);

/// \brief Allocate an external pointer to an ArrowArray
///
/// Allocate an external pointer to an uninitialized ArrowArray with a finalizer that
/// ensures that any non-null release callback in a pointed-to structure will be called
/// when the external pointer is garbage collected.
static inline SEXP nanoarrow_array_owning_xptr(void);

/// \brief Allocate an external pointer to an ArrowArrayStream
///
/// Allocate an external pointer to an uninitialized ArrowArrayStream with a finalizer
/// that ensures that any non-null release callback in a pointed-to structure will be
/// called when the external pointer is garbage collected.
static inline SEXP nanoarrow_array_stream_owning_xptr(void);

/// \brief Ensure an input SEXP points to an initialized ArrowSchema
///
/// This function will always return an ArrowSchema pointer that can be safely
/// consumed or raise an error via Rf_error(). This is intended to be used to
/// sanitize an *input* ArrowSchema.
static inline struct ArrowSchema* nanoarrow_schema_from_xptr(SEXP schema_xptr);

/// \brief Ensure an output SEXP points to an uninitialized ArrowSchema
///
/// This function will always return an ArrowSchema pointer that can be safely
/// used as an output argument or raise an error via Rf_error(). This is intended
/// to be used to sanitize an *output* ArrowSchema allocated from R or elsewhere.
static inline struct ArrowSchema* nanoarrow_output_schema_from_xptr(SEXP schema_xptr);

/// \brief Ensure an input SEXP points to an initialized ArrowArray
///
/// This function will always return an ArrowArray pointer that can be safely
/// consumed or raise an error via Rf_error(). This is intended to be used to
/// sanitize an *input* ArrowArray.
static inline struct ArrowArray* nanoarrow_array_from_xptr(SEXP array_xptr);

/// \brief Ensure an output SEXP points to an uninitialized ArrowArray
///
/// This function will always return an ArrowArray pointer that can be safely
/// used as an output argument or raise an error via Rf_error(). This is intended
/// to be used to sanitize an *output* ArrowArray allocated from R or elsewhere.
static inline struct ArrowArray* nanoarrow_output_array_from_xptr(SEXP array_xptr);

/// \brief Ensure an input SEXP points to an initialized ArrowArrayStream
///
/// This function will always return an ArrowArrayStream pointer that can be safely
/// consumed or raise an error via Rf_error(). This is intended to be used to
/// sanitize an *input* ArrowArrayStream.
static inline struct ArrowArrayStream* nanoarrow_array_stream_from_xptr(
    SEXP array_stream_xptr);

/// \brief Ensure an output SEXP points to an uninitialized ArrowArrayStream
///
/// This function will always return an ArrowArrayStream pointer that can be safely
/// used as an output argument or raise an error via Rf_error(). This is intended
/// to be used to sanitize an *output* ArrowArrayStream allocated from R or elsewhere.
static inline struct ArrowArrayStream* nanoarrow_output_array_stream_from_xptr(
    SEXP array_stream_xptr);

/// \brief Finalize an external pointer to an ArrowSchema
///
/// This function is provided for internal use by nanoarrow_schema_owning_xptr()
/// and should not be called directly.
static void nanoarrow_finalize_schema_xptr(SEXP schema_xptr);

/// \brief Finalize an external pointer to an ArrowArray
///
/// This function is provided for internal use by nanoarrow_array_owning_xptr()
/// and should not be called directly.
static void nanoarrow_finalize_array_xptr(SEXP array_xptr);

/// \brief Finalize an external pointer to an ArrowArrayStream
///
/// This function is provided for internal use by nanoarrow_array_stream_owning_xptr()
/// and should not be called directly.
static void nanoarrow_finalize_array_stream_xptr(SEXP array_stream_xptr);

/// @}

// Implementations follow

static void nanoarrow_finalize_schema_xptr(SEXP schema_xptr) {
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema != NULL && schema->release != NULL) {
    schema->release(schema);
  }

  if (schema != NULL) {
    free(schema);
    R_ClearExternalPtr(schema_xptr);
  }
}

static void nanoarrow_finalize_array_xptr(SEXP array_xptr) {
  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array != NULL && array->release != NULL) {
    array->release(array);
  }

  if (array != NULL) {
    free(array);
    R_ClearExternalPtr(array_xptr);
  }
}

static void nanoarrow_finalize_array_stream_xptr(SEXP array_stream_xptr) {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream != NULL && array_stream->release != NULL) {
    array_stream->release(array_stream);
  }

  if (array_stream != NULL) {
    free(array_stream);
    R_ClearExternalPtr(array_stream_xptr);
  }
}

static inline SEXP nanoarrow_schema_owning_xptr(void) {
  struct ArrowSchema* schema = (struct ArrowSchema*)malloc(sizeof(struct ArrowSchema));
  if (schema == NULL) {
    Rf_error("Failed to allocate ArrowSchema");
  }

  schema->release = NULL;

  SEXP schema_xptr = PROTECT(R_MakeExternalPtr(schema, R_NilValue, R_NilValue));
  SEXP schema_cls = PROTECT(Rf_mkString("nanoarrow_schema"));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, schema_cls);
  R_RegisterCFinalizer(schema_xptr, &nanoarrow_finalize_schema_xptr);
  UNPROTECT(2);
  return schema_xptr;
}

static inline SEXP nanoarrow_array_owning_xptr(void) {
  struct ArrowArray* array = (struct ArrowArray*)malloc(sizeof(struct ArrowArray));
  array->release = NULL;

  SEXP array_xptr = PROTECT(R_MakeExternalPtr(array, R_NilValue, R_NilValue));
  SEXP array_cls = PROTECT(Rf_mkString("nanoarrow_array"));
  Rf_setAttrib(array_xptr, R_ClassSymbol, array_cls);
  R_RegisterCFinalizer(array_xptr, &nanoarrow_finalize_array_xptr);
  UNPROTECT(2);
  return array_xptr;
}

static inline SEXP nanoarrow_array_stream_owning_xptr(void) {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)malloc(sizeof(struct ArrowArrayStream));
  array_stream->release = NULL;

  SEXP array_stream_xptr =
      PROTECT(R_MakeExternalPtr(array_stream, R_NilValue, R_NilValue));
  SEXP array_stream_cls = PROTECT(Rf_mkString("nanoarrow_array_stream"));
  Rf_setAttrib(array_stream_xptr, R_ClassSymbol, array_stream_cls);
  R_RegisterCFinalizer(array_stream_xptr, &nanoarrow_finalize_array_stream_xptr);
  UNPROTECT(2);
  return array_stream_xptr;
}

static inline struct ArrowSchema* nanoarrow_schema_from_xptr(SEXP schema_xptr) {
  if (!Rf_inherits(schema_xptr, "nanoarrow_schema")) {
    Rf_error("`schema` argument that does not inherit from 'nanoarrow_schema'");
  }

  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema == NULL) {
    Rf_error("nanoarrow_schema() is an external pointer to NULL");
  }

  if (schema->release == NULL) {
    Rf_error("nanoarrow_schema() has already been released");
  }

  return schema;
}

static inline struct ArrowSchema* nanoarrow_output_schema_from_xptr(SEXP schema_xptr) {
  if (!Rf_inherits(schema_xptr, "nanoarrow_schema")) {
    Rf_error("`schema` argument that does not inherit from 'nanoarrow_schema'");
  }

  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema == NULL) {
    Rf_error("nanoarrow_schema() is an external pointer to NULL");
  }

  if (schema->release != NULL) {
    Rf_error("nanoarrow_schema() output has already been initialized");
  }

  return schema;
}

static inline struct ArrowArray* nanoarrow_array_from_xptr(SEXP array_xptr) {
  if (!Rf_inherits(array_xptr, "nanoarrow_array")) {
    Rf_error("`array` argument that is not a nanoarrow_array()");
  }

  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array == NULL) {
    Rf_error("nanoarrow_array() is an external pointer to NULL");
  }

  if (array->release == NULL) {
    Rf_error("nanoarrow_array() has already been released");
  }

  return array;
}

static inline struct ArrowArray* nanoarrow_output_array_from_xptr(SEXP array_xptr) {
  if (!Rf_inherits(array_xptr, "nanoarrow_array")) {
    Rf_error("`array` argument that is not a nanoarrow_array()");
  }

  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array == NULL) {
    Rf_error("nanoarrow_array() is an external pointer to NULL");
  }

  if (array->release != NULL) {
    Rf_error("nanoarrow_array() output has already been initialized");
  }

  return array;
}

static inline struct ArrowArrayStream* nanoarrow_array_stream_from_xptr(
    SEXP array_stream_xptr) {
  if (!Rf_inherits(array_stream_xptr, "nanoarrow_array_stream")) {
    Rf_error("`array_stream` argument that is not a nanoarrow_array_stream()");
  }

  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream == NULL) {
    Rf_error("nanoarrow_array_stream() is an external pointer to NULL");
  }

  if (array_stream->release == NULL) {
    Rf_error("nanoarrow_array_stream() has already been released");
  }

  return array_stream;
}

static inline struct ArrowArrayStream* nanoarrow_output_array_stream_from_xptr(
    SEXP array_stream_xptr) {
  if (!Rf_inherits(array_stream_xptr, "nanoarrow_array_stream")) {
    Rf_error("`array_stream` argument that is not a nanoarrow_array_stream()");
  }

  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream == NULL) {
    Rf_error("nanoarrow_array_stream() is an external pointer to NULL");
  }

  if (array_stream->release != NULL) {
    Rf_error("nanoarrow_array_stream() output has already been initialized");
  }

  return array_stream;
}

#ifdef __cplusplus
}
#endif

#endif
