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

#ifndef R_NANOARROW_BUFFER_H_INCLUDED
#define R_NANOARROW_BUFFER_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"
#include "preserve.h"
#include "util.h"

void finalize_buffer_xptr(SEXP buffer_xptr);
void nanoarrow_sexp_deallocator(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                int64_t size);

// Create an external pointer with the proper class and that will release any
// non-null, non-released pointer when garbage collected.
static inline SEXP buffer_owning_xptr(void) {
  struct ArrowBuffer* buffer =
      (struct ArrowBuffer*)ArrowMalloc(sizeof(struct ArrowBuffer));
  ArrowBufferInit(buffer);

  SEXP buffer_xptr = PROTECT(R_MakeExternalPtr(buffer, R_NilValue, R_NilValue));
  Rf_setAttrib(buffer_xptr, R_ClassSymbol, nanoarrow_cls_buffer);
  R_RegisterCFinalizer(buffer_xptr, &finalize_buffer_xptr);
  UNPROTECT(1);
  return buffer_xptr;
}

// Create an arrow_buffer with a deallocator that will release shelter when
// the buffer is no longer needed.
static inline void buffer_borrowed(struct ArrowBuffer* buffer, const void* addr,
                                   int64_t size_bytes, SEXP shelter) {
  buffer->allocator = ArrowBufferDeallocator(&nanoarrow_sexp_deallocator, shelter);
  buffer->data = (uint8_t*)addr;
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = size_bytes;
  nanoarrow_preserve_sexp(shelter);
}

static inline SEXP buffer_borrowed_xptr(const void* addr, int64_t size_bytes,
                                        SEXP shelter) {
  SEXP buffer_xptr = PROTECT(buffer_owning_xptr());

  // Don't bother with a preserve/release if the buffer is NULL
  if (addr == NULL) {
    UNPROTECT(1);
    return buffer_xptr;
  }

  struct ArrowBuffer* buffer = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);
  buffer_borrowed(buffer, addr, size_bytes, shelter);
  UNPROTECT(1);
  return buffer_xptr;
}

static inline void buffer_borrowed_xptr_set_type(SEXP buffer_xptr,
                                                 enum ArrowBufferType buffer_type,
                                                 enum ArrowType buffer_data_type,
                                                 int64_t element_size_bits) {
  SEXP buffer_types_sexp = PROTECT(Rf_allocVector(INTSXP, 3));
  INTEGER(buffer_types_sexp)[0] = buffer_type;
  INTEGER(buffer_types_sexp)[1] = buffer_data_type;
  INTEGER(buffer_types_sexp)[2] = (int32_t)element_size_bits;
  R_SetExternalPtrTag(buffer_xptr, buffer_types_sexp);
  UNPROTECT(1);
}

static inline struct ArrowBuffer* buffer_from_xptr(SEXP buffer_xptr) {
  if (!Rf_inherits(buffer_xptr, "nanoarrow_buffer")) {
    Rf_error("`buffer` argument that is not a nanoarrow_buffer()");
  }

  struct ArrowBuffer* buffer = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);
  if (buffer == NULL) {
    Rf_error("nanoarrow_buffer is an external pointer to NULL");
  }

  return buffer;
}

#endif
