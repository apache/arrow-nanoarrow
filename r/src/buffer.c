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

SEXP nanoarrow_c_buffer_info(SEXP buffer_xptr) {
  return R_ExternalPtrTag(buffer_xptr);
}

SEXP nanoarrow_c_buffer_as_raw(SEXP buffer_xptr) {
  SEXP info = R_ExternalPtrTag(buffer_xptr);
  if (info == R_NilValue) {
    Rf_error("Can't as.raw() a nanoarrow_buffer with unknown size");
  }

  SEXP size_bytes_sexp = VECTOR_ELT(info, 0);
  R_xlen_t size_bytes = 0;
  if (TYPEOF(size_bytes_sexp) == INTSXP) {
    size_bytes = INTEGER(size_bytes_sexp)[0];
  } else if (TYPEOF(size_bytes_sexp) == REALSXP) {
    size_bytes = REAL(size_bytes_sexp)[0];
  } else {
    Rf_error("Unknown object type for nanoarrow_buffer size_bytes");
  }

  SEXP result = PROTECT(Rf_allocVector(RAWSXP, size_bytes));
  memcpy(RAW(result), R_ExternalPtrAddr(buffer_xptr), size_bytes);
  UNPROTECT(1);
  return result;
}
