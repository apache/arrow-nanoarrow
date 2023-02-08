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

#include <string.h>

#include "buffer.h"
#include "nanoarrow.h"

void finalize_buffer_xptr(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = (struct ArrowBuffer*)R_ExternalPtrAddr(buffer_xptr);
  if (buffer != NULL) {
    ArrowBufferReset(buffer);
  }
}

void nanoarrow_sexp_deallocator(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                int64_t size) {
  R_ReleaseObject((SEXP)allocator->private_data);
}

SEXP nanoarrow_c_buffer_info(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);

  const char* names[] = {"data", "size_bytes", "capacity_bytes", ""};
  SEXP info = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(info, 0, R_MakeExternalPtr(buffer->data, NULL, buffer_xptr));
  SET_VECTOR_ELT(info, 1, Rf_ScalarReal(buffer->size_bytes));
  SET_VECTOR_ELT(info, 2, Rf_ScalarReal(buffer->capacity_bytes));
  UNPROTECT(1);
  return info;
}

SEXP nanoarrow_c_buffer_as_raw(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);

  SEXP result = PROTECT(Rf_allocVector(RAWSXP, buffer->size_bytes));
  memcpy(RAW(result), buffer->data, buffer->size_bytes);
  UNPROTECT(1);
  return result;
}
