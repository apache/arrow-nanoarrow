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

#include "materialize_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// A heuristic to identify prototypes that should be treated like data frames
// (i.e., including record-style vectors like POSIXct). This heuristic returns
// true if ptype is a data.frame or is an S3 list with names.
int nanoarrow_ptype_is_data_frame(SEXP ptype);

// Returns the number of rows in a data.frame in a way that is least likely to
// expand the attr(x, "row.names")
R_xlen_t nanoarrow_data_frame_size(SEXP x);

// Set rownames of a data.frame (with special handling if len > INT_MAX)
void nanoarrow_set_rownames(SEXP x, R_xlen_t len);

// Perform actual materializing of values (e.g., loop through buffers)
int nanoarrow_materialize(struct RConverter* converter, SEXP converter_xptr);

// Shortcut to allocate a vector based on a vector type or ptype
SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len);
SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len);

// Finalize an object before returning to R. Currently only used for
// nanoarrow_vctr conversion.
int nanoarrow_materialize_finalize_result(SEXP converter_xptr);

#ifdef __cplusplus
}
#endif

#endif
