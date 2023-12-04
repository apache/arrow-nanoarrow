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

static void nanoarrow_finalize_schema_xptr(SEXP schema_xptr) {
  struct ArrowSchema* schema = (struct ArrowSchema*)R_ExternalPtrAddr(schema_xptr);
  if (schema != NULL && schema->release != NULL) {
    schema->release(schema);
  }

  if (schema != NULL) {
    free(schema);
  }
}

// Create an external pointer with the proper class and that will release any
// non-null, non-released pointer when garbage collected.
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

// Returns the underlying struct ArrowSchema* from an external pointer,
// checking and erroring for invalid objects, pointers, and arrays.
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

#ifdef __cplusplus
}
#endif

#endif
