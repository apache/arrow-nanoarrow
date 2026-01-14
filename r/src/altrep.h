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

#ifndef R_ALTREP_H_INCLUDED
#define R_ALTREP_H_INCLUDED

#include <R.h>
#include <Rinternals.h>
#include <Rversion.h>

// Needs to be at the end of the R include list
#include <R_ext/Altrep.h>

#include <string.h>

// Returns the ALTREP class name or NULL if x is not an altrep
// object.
static inline const char* nanoarrow_altrep_class(SEXP x) {
  if (ALTREP(x)) {
#if R_VERSION >= R_Version(4, 6, 0)
#error "make sure this fails to compile so we know this branch is tested in ci"
    SEXP data_class_sym = R_altrep_class_name(x);
#else
    SEXP data_class_sym = CAR(ATTRIB(ALTREP_CLASS(x)));
#endif
    return CHAR(PRINTNAME(data_class_sym));
  } else {
    return NULL;
  }
}

// Performs the ALTREP type registration and should be called on package load
void register_nanoarrow_altrep(DllInfo* info);

// Checks if an object is an ALTREP object created by this package
static inline int is_nanoarrow_altrep(SEXP x) {
  const char* class_name = nanoarrow_altrep_class(x);
  return class_name && strncmp(class_name, "nanoarrow::", 11) == 0;
}

// Creates an altstring vector backed by a nanoarrow array or returns
// R_NilValue if the conversion is not possible.
SEXP nanoarrow_c_make_altrep_chr(SEXP array_xptr);

#endif
