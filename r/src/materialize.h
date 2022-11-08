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

// Perform actual materializing of values (e.g., loop through buffers)
int nanoarrow_materialize(struct RConverter* converter);

// Shortcut to allocate a vector based on a vector type or ptype
SEXP nanoarrow_alloc_type(enum VectorType vector_type, R_xlen_t len);
SEXP nanoarrow_materialize_realloc(SEXP ptype, R_xlen_t len);

#endif
