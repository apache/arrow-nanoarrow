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

#ifndef R_MATERIALIZE_H_INCLUDED
#define R_MATERIALIZE_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

// Vector types that have some special casing internally to avoid unnecessary allocations
// or looping at the R level. Other types are represented by an SEXP ptype.
enum VectorType {
  VECTOR_TYPE_NULL,
  VECTOR_TYPE_UNSPECIFIED,
  VECTOR_TYPE_LGL,
  VECTOR_TYPE_INT,
  VECTOR_TYPE_DBL,
  VECTOR_TYPE_ALTREP_CHR,
  VECTOR_TYPE_CHR,
  VECTOR_TYPE_POSIXCT,
  VECTOR_TYPE_DATE,
  VECTOR_TYPE_DIFFTIME,
  VECTOR_TYPE_BLOB,
  VECTOR_TYPE_LIST_OF,
  VECTOR_TYPE_DATA_FRAME,
  VECTOR_TYPE_OTHER
};

enum RTimeUnits {
  R_TIME_UNIT_SECONDS,
  R_TIME_UNIT_MINUTES,
  R_TIME_UNIT_HOURS,
  R_TIME_UNIT_DAYS,
  R_TIME_UNIT_WEEKS
};

struct PTypeView {
  enum VectorType vector_type;
  int sexp_type;
  enum RTimeUnits r_time_units;
  SEXP ptype;
};

// A wrapper around the ArrayView with an additional offset + length
// representing a source of a materialization
struct ArrayViewSlice {
  struct ArrowSchemaView schema_view;
  struct ArrowArrayView* array_view;
  int64_t offset;
  int64_t length;
};

// A wapper around an SEXP vector with an additional offset + length.
// This can be both a source and/or a target for copying from/to.
struct VectorSlice {
  SEXP vec_sexp;
  R_xlen_t offset;
  R_xlen_t length;
};

// Options for resolving a ptype and for materializing values. These are
// currently unused.
struct MaterializeOptions {
  double scale;
};

struct RConverter {
  struct PTypeView ptype_view;
  struct ArrowSchemaView schema_view;
  struct ArrowArrayView array_view;
  struct ArrayViewSlice src;
  struct VectorSlice dst;
  struct MaterializeOptions* options;
  struct ArrowError error;
  R_xlen_t size;
  R_xlen_t capacity;
};


SEXP nanoarrow_converter_from_type(enum VectorType vector_type);
SEXP nanoarrow_converter_from_ptype(SEXP ptype);
int nanoarrow_converter_set_schema(SEXP converter_xptr, SEXP schema_xptr);
int nanoarrow_converter_set_array(SEXP converter_xptr, SEXP array_xptr);
int nanoarrow_converter_reserve(SEXP converter_xptr, R_xlen_t additional_size);
R_xlen_t nanoarrow_converter_materialize_n(SEXP converter_xptr, R_xlen_t n);
int nanoarrow_converter_materialize_all(SEXP converter_xptr);
int nanoarrow_converter_finalize(SEXP converter_xptr);
SEXP nanoarrow_converter_result(SEXP converter_xptr);


static inline struct ArrayViewSlice DefaultArrayViewSlice(
    struct ArrowArrayView* array_view) {
  struct ArrayViewSlice slice;
  slice.schema_view.data_type = NANOARROW_TYPE_UNINITIALIZED;
  slice.schema_view.storage_data_type = NANOARROW_TYPE_UNINITIALIZED;
  slice.array_view = array_view;
  slice.offset = 0;
  slice.length = array_view->array->length;
  return slice;
}

static inline struct VectorSlice DefaultVectorSlice(SEXP vec_sexp) {
  struct VectorSlice slice;
  slice.vec_sexp = vec_sexp;
  slice.offset = 0;
  slice.length = Rf_xlength(vec_sexp);
  return slice;
}

static inline struct MaterializeOptions DefaultMaterializeOptions() {
  struct MaterializeOptions options;
  options.scale = 1;
  return options;
}

// Utility for "length" in the context of a materialized value.
// This is the same as vctrs::vec_size(): Rf_xlength() for vectors and
// the number of rows for data.frame()/matrix().
R_xlen_t nanoarrow_vec_size(SEXP vec_sexp);

// Reallocate ptype to a given length. Currently only zero-size ptypes
// are supported but in the future this could also copy existing values
// from ptype to provide growable behaviour.
SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len);

// A shortuct version of nanoarrow_materialize_realloc() that doesn't require
// allocating a ptype first.
SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len);

// This function populates the given VectorSlice from the ArrayViewSlice,
// returning 0 on success or something else if the type conversion is not
// possible.
int nanoarrow_materialize(struct ArrayViewSlice* src, struct VectorSlice* dst,
                          struct MaterializeOptions* options);

// This function populates a VectorSlice from a VectorSlice (i.e., copies existing
// values). This is used to support calling into the R-level materialize_array()
// from nanoarrow_materialize() when we have already preallocated a result.
// This is similar to rlang::vec_poke_range() except it also supports matrices.
int nanoarrow_copy_vector(struct VectorSlice* src, struct VectorSlice* dst,
                          struct MaterializeOptions* options);

// This function mus be called after an alloc + zero or more materialize calls to finalize
// the R object. Note that the returned SEXP can be different than result_sexp (e.g., if
// the result of finalizing required a reallocation). Currently this function just checks
// that the value of len and returns result_sexp.
SEXP nanoarrow_materialize_finish(SEXP result_sexp, R_xlen_t len,
                                  struct MaterializeOptions* options);

#endif
