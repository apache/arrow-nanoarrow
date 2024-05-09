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

#include "nanoarrow.h"

SEXP nanoarrow_c_vctr_chunk_offsets(SEXP array_list) {
  int num_chunks = Rf_length(array_list);
  SEXP offsets_sexp = PROTECT(Rf_allocVector(INTSXP, num_chunks + 1));
  int* offsets = INTEGER(offsets_sexp);
  offsets[0] = 0;
  int64_t cumulative_offset = 0;

  struct ArrowArray* array;
  for (int i = 0; i < num_chunks; i++) {
    array = (struct ArrowArray*)R_ExternalPtrAddr(VECTOR_ELT(array_list, i));
    cumulative_offset += array->length;
    if (cumulative_offset > INT_MAX) {
      Rf_error("Can't build nanoarrow_vctr with length > INT_MAX");  // # nocov
    }

    offsets[i + 1] = cumulative_offset;
  }

  UNPROTECT(1);
  return offsets_sexp;
}

static int resolve_chunk(int* sorted_offsets, int index, int start_offset_i,
                         int end_offset_i) {
  if (start_offset_i >= (end_offset_i - 1)) {
    return start_offset_i;
  }

  int mid_offset_i = start_offset_i + (end_offset_i - start_offset_i) / 2;
  int mid_index = sorted_offsets[mid_offset_i];
  if (index < mid_index) {
    return resolve_chunk(sorted_offsets, index, start_offset_i, mid_offset_i);
  } else {
    return resolve_chunk(sorted_offsets, index, mid_offset_i, end_offset_i);
  }
}

SEXP nanoarrow_c_vctr_chunk_resolve(SEXP indices_sexp, SEXP offsets_sexp) {
  int* offsets = INTEGER(offsets_sexp);
  int end_offset_i = Rf_length(offsets_sexp) - 1;
  int last_offset = offsets[end_offset_i];

  int n = Rf_length(indices_sexp);
  SEXP chunk_indices_sexp = PROTECT(Rf_allocVector(INTSXP, n));
  int* chunk_indices = INTEGER(chunk_indices_sexp);

  int buf[1024];
  for (int i = 0; i < n; i++) {
    if (i % 1024 == 0) {
      INTEGER_GET_REGION(indices_sexp, i, 1024, buf);
    }
    int index0 = buf[i % 1024];

    if (index0 < 0 || index0 > last_offset) {
      chunk_indices[i] = NA_INTEGER;
    } else {
      chunk_indices[i] = resolve_chunk(offsets, index0, 0, end_offset_i);
    }
  }

  UNPROTECT(1);
  return chunk_indices_sexp;
}

SEXP nanoarrow_c_vctr_as_slice(SEXP indices_sexp) {
  if (TYPEOF(indices_sexp) != INTSXP) {
    return R_NilValue;
  }
  SEXP slice_sexp = PROTECT(Rf_allocVector(INTSXP, 2));
  int* slice = INTEGER(slice_sexp);

  int n = Rf_length(indices_sexp);
  slice[1] = n;

  if (n == 1) {
    slice[0] = INTEGER_ELT(indices_sexp, 0);
    UNPROTECT(1);
    return slice_sexp;
  } else if (n == 0) {
    slice[0] = NA_INTEGER;
    UNPROTECT(1);
    return slice_sexp;
  }

  int buf[1024];
  INTEGER_GET_REGION(indices_sexp, 0, 1024, buf);
  slice[0] = buf[0];

  int last_value = buf[0];
  int this_value = 0;

  for (int i = 1; i < n; i++) {
    if (i % 1024 == 0) {
      INTEGER_GET_REGION(indices_sexp, i, 1024, buf);
    }

    this_value = buf[i % 1024];
    if ((this_value - last_value) != 1) {
      UNPROTECT(1);
      return R_NilValue;
    }

    last_value = this_value;
  }

  UNPROTECT(1);
  return slice_sexp;
}
