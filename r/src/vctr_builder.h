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

#ifndef R_NANOARROW_VCTR_BUILDER_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#ifdef __cplusplus
extern "C" {
#endif

// An opaque pointer to a C++-implemented collector that is instantiated
// using an ArrowSchema and (optionally) a SEXP ptype and is supplied
// zero or more ArrowArrays before computing the output.
struct VctrBuilder;

// Options for when to use ALTREP. Currently ALTREP is only implemented
// for character() with exactly one input chunk. The default may eventually
// use some heuristics to decide if there is a likely performance advantage
// to deferring the conversion.
enum VctrBuilderUseAltrep {
  VCTR_BUILDER_USE_ALTREP_DEFAULT = 0,
  VCTR_BUILDER_USE_ALTREP_ALWAYS = 1,
  VCTR_BUILDER_USE_ALTREP_NEVER = 2
};

// Options controlling the details of how arrays are built. Note that
// this does not control the destination ptype: customizing ptype resolution
// is currently possible by passing a function to the `to` argument at the
// top level. Future additions could control the error/warning strategy
// for (potentially) lossy conversions.
struct VctrBuilderOptions {
  enum VctrBuilderUseAltrep use_altrep;
};

SEXP nanoarrow_vctr_builder_init(SEXP schema_xptr, SEXP ptype_sexp);

SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr);

SEXP nanoarrow_c_convert_array2(SEXP array_xptr, SEXP ptype_sexp);

#ifdef __cplusplus
}
#endif

#endif
