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

#ifndef R_CONVERT_H_INCLUDED
#define R_CONVERT_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

#include "materialize.h"

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
SEXP nanoarrow_converter_release_result(SEXP converter_xptr);

// Calls Rf_error() with the internal error buffer populated by above calls
// that return a non-zero errno value.
void nanoarrow_converter_stop(SEXP converter_xptr);

#endif
