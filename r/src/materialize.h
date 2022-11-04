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
// or looping at the R level. Some of these types also need an SEXP ptype to communicate
// additional information.
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

// More easily switch()able version of attr(difftime_obj, "units")
enum RTimeUnits {
  R_TIME_UNIT_SECONDS,
  R_TIME_UNIT_MINUTES,
  R_TIME_UNIT_HOURS,
  R_TIME_UNIT_DAYS,
  R_TIME_UNIT_WEEKS
};

// A "parsed" version of an SEXP ptype (like a SchemaView but for
// R objects))
struct PTypeView {
  enum VectorType vector_type;
  int sexp_type;
  enum RTimeUnits r_time_units;
  SEXP ptype;
};

// A wrapper around the ArrayView with an additional offset + length
// representing a source of a materialization
struct ArrayViewSlice {
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
// currently unused but this struct is a placeholder for them when they
// are implemented.
struct MaterializeOptions {
  double scale;
};

// A house for a conversion operation (i.e., zero or more arrays
// getting converted into an R vector)). The structure of this
// may change in the future but the API below should be relatively stable.
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
  R_xlen_t n_children;
  struct RConverter** children;
};

// Create and initialize a converter. A converter's output R vector type
// never changes once it has been created.
SEXP nanoarrow_converter_from_type(enum VectorType vector_type);
SEXP nanoarrow_converter_from_ptype(SEXP ptype);

// Set the schema for the next array that will be materialized into
// the R vector. In theory this could change although this has not been
// implemented. This will also validate the schema. Returns an errno code.
int nanoarrow_converter_set_schema(SEXP converter_xptr, SEXP schema_xptr);

// Set the array target. This will also validate the array against the last
// schema that was set. Returns an errno code.
int nanoarrow_converter_set_array(SEXP converter_xptr, SEXP array_xptr);

// Reserve space in the R vector output for additional elements. In theory
// this could be used to provide growable behaviour; however, this is not
// implemented. Returns an errno code.
int nanoarrow_converter_reserve(SEXP converter_xptr, R_xlen_t additional_size);

// Materialize the next n elements into the output. Returns the number of elements
// that were actualy materialized which may be less than n.
R_xlen_t nanoarrow_converter_materialize_n(SEXP converter_xptr, R_xlen_t n);

// Materialize the entire array into the output. Returns an errno code.
int nanoarrow_converter_materialize_all(SEXP converter_xptr);

// Finalize the output. Currently this just validates the length of the
// output. Returns an errno code.
int nanoarrow_converter_finalize(SEXP converter_xptr);

// Returns the resulting SEXP and moves the result out of the protection
// of the converter.
SEXP nanoarrow_converter_result(SEXP converter_xptr);

// Calls Rf_error() with the internal error buffer populated by above calls
// that return a non-zero errno value.
void nanoarrow_converter_stop(SEXP converter_xptr);

// Shortcut to allocate a vector based on a vector type. This is used in
// infer_ptype.c.
SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len);

#endif
