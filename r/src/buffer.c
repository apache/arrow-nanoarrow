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
    ArrowFree(buffer);
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
  int32_t element_size_bits = 0;
  enum ArrowType buffer_data_type = NANOARROW_TYPE_UNINITIALIZED;

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
      } else {
        Rf_error("NA_character_ not supported in as_nanoarrow_buffer()");
      }
      break;
    default:
      Rf_error("Unsupported type");
  }

  switch (TYPEOF(x_sexp)) {
    case NILSXP:
    case RAWSXP:
      buffer_data_type = NANOARROW_TYPE_BINARY;
      size_bytes = len;
      element_size_bits = 8;
      break;
    case LGLSXP:
    case INTSXP:
      buffer_data_type = NANOARROW_TYPE_INT32;
      size_bytes = len * sizeof(int);
      element_size_bits = 8 * sizeof(int);
      break;
    case REALSXP:
      buffer_data_type = NANOARROW_TYPE_DOUBLE;
      size_bytes = len * sizeof(double);
      element_size_bits = 8 * sizeof(double);
      break;
    case CPLXSXP:
      buffer_data_type = NANOARROW_TYPE_DOUBLE;
      size_bytes = len * 2 * sizeof(double);
      element_size_bits = 8 * sizeof(double);
      break;
    case CHARSXP:
      buffer_data_type = NANOARROW_TYPE_STRING;
      size_bytes = Rf_xlength(x_sexp);
      element_size_bits = 8;
      break;
    default:
      break;
  }

  // Don't bother borrowing a zero-size buffer
  SEXP buffer_xptr;
  if (size_bytes == 0) {
    buffer_xptr = PROTECT(buffer_owning_xptr());
  } else {
    buffer_xptr = PROTECT(buffer_borrowed_xptr(data, size_bytes, x_sexp));
  }

  buffer_borrowed_xptr_set_type(buffer_xptr, NANOARROW_BUFFER_TYPE_DATA, buffer_data_type,
                                element_size_bits);

  UNPROTECT(1);
  return buffer_xptr;
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

  SEXP buffer_types_sexp = R_ExternalPtrTag(buffer_xptr);

  SEXP buffer_type_sexp;
  SEXP buffer_data_type_sexp;
  int32_t element_size_bits;

  if (buffer_types_sexp == R_NilValue) {
    buffer_type_sexp = PROTECT(Rf_mkString("unknown"));
    buffer_data_type_sexp = PROTECT(Rf_mkString("unknown"));
    element_size_bits = 0;
  } else {
    enum ArrowBufferType buffer_type = INTEGER(buffer_types_sexp)[0];
    const char* buffer_type_string;
    switch (buffer_type) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        buffer_type_string = "validity";
        break;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        buffer_type_string = "data_offset";
        break;
      case NANOARROW_BUFFER_TYPE_DATA:
        buffer_type_string = "data";
        break;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
        buffer_type_string = "type_id";
        break;
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        buffer_type_string = "union_offset";
        break;
      case NANOARROW_BUFFER_TYPE_VARIADIC_DATA:
        buffer_type_string = "variadic_data";
        break;
      case NANOARROW_BUFFER_TYPE_VARIADIC_SIZE:
        buffer_type_string = "variadic_size";
        break;
      default:
        buffer_type_string = "unknown";
        break;
    }

    enum ArrowType buffer_data_type = INTEGER(buffer_types_sexp)[1];
    const char* buffer_data_type_string = ArrowTypeString(buffer_data_type);

    buffer_type_sexp = PROTECT(Rf_mkString(buffer_type_string));
    buffer_data_type_sexp = PROTECT(Rf_mkString(buffer_data_type_string));
    element_size_bits = INTEGER(buffer_types_sexp)[2];
  }

  const char* names[] = {"data", "size_bytes", "capacity_bytes",
                         "type", "data_type",  "element_size_bits",
                         ""};
  SEXP info = PROTECT(Rf_mkNamed(VECSXP, names));
  SET_VECTOR_ELT(info, 0, R_MakeExternalPtr(buffer->data, R_NilValue, buffer_xptr));
  SET_VECTOR_ELT(info, 1, Rf_ScalarReal((double)buffer->size_bytes));
  SET_VECTOR_ELT(info, 2, Rf_ScalarReal((double)buffer->capacity_bytes));
  SET_VECTOR_ELT(info, 3, buffer_type_sexp);
  SET_VECTOR_ELT(info, 4, buffer_data_type_sexp);
  SET_VECTOR_ELT(info, 5, Rf_ScalarInteger(element_size_bits));
  UNPROTECT(3);
  return info;
}

SEXP nanoarrow_c_buffer_head_bytes(SEXP buffer_xptr, SEXP max_bytes_sexp) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);
  int64_t max_bytes = (int64_t)REAL(max_bytes_sexp)[0];

  if (buffer->size_bytes <= max_bytes) {
    return buffer_xptr;
  }

  SEXP buffer_clone_xptr =
      PROTECT(buffer_borrowed_xptr(buffer->data, max_bytes, buffer_xptr));
  R_SetExternalPtrTag(buffer_clone_xptr, Rf_duplicate(R_ExternalPtrTag(buffer_xptr)));
  UNPROTECT(1);
  return buffer_clone_xptr;
}

SEXP nanoarrow_c_buffer_as_raw(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);

  SEXP result = PROTECT(Rf_allocVector(RAWSXP, buffer->size_bytes));
  if (buffer->size_bytes > 0) {
    memcpy(RAW(result), buffer->data, buffer->size_bytes);
  }
  UNPROTECT(1);
  return result;
}
