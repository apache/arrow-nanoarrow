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
  nanoarrow_release_sexp((SEXP)allocator->private_data);
}

SEXP nanoarrow_c_as_buffer_default(SEXP x_sexp) {
  R_xlen_t len = Rf_xlength(x_sexp);
  const void* data = NULL;
  int64_t size_bytes = 0;

  // For non-NA character(1), we use the first element
  if (TYPEOF(x_sexp) == STRSXP && len == 1) {
    return nanoarrow_c_as_buffer_default(STRING_ELT(x_sexp, 0));
  }

  switch (TYPEOF(x_sexp)) {
    case NILSXP:
      data = NULL;
      break;
    case RAWSXP:
    case LGLSXP:
    case INTSXP:
    case REALSXP:
    case CPLXSXP:
      data = DATAPTR_RO(x_sexp);
      break;
    case CHARSXP:
      if (x_sexp != NA_STRING) {
        data = CHAR(x_sexp);
        break;
      }
    default:
      Rf_error("Unsupported type");
  }

  switch (TYPEOF(x_sexp)) {
    case NILSXP:
    case RAWSXP:
      size_bytes = len;
      break;
    case LGLSXP:
    case INTSXP:
      size_bytes = len * sizeof(int);
      break;
    case REALSXP:
      size_bytes = len * sizeof(double);
      break;
    case CPLXSXP:
      size_bytes = len * 2 * sizeof(double);
      break;
    case CHARSXP:
      size_bytes = Rf_xlength(x_sexp);
      break;
    default:
      break;
  }

  // Don't bother borrowing a zero-size buffer
  if (size_bytes == 0) {
    return buffer_owning_xptr();
  } else {
    return buffer_borrowed_xptr(data, size_bytes, x_sexp);
  }
}

SEXP nanoarrow_c_buffer_append(SEXP buffer_xptr, SEXP new_buffer_xptr) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);
  struct ArrowBuffer* new_buffer = buffer_from_xptr(new_buffer_xptr);

  int result = ArrowBufferAppend(buffer, new_buffer->data, new_buffer->size_bytes);
  if (result != NANOARROW_OK) {
    Rf_error("ArrowBufferAppend() failed");
  }

  return R_NilValue;
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
