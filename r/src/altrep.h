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

#include "Rversion.h"

#include <string.h>

// ALTREP available in R >= 3.5
#if defined(R_VERSION) && R_VERSION >= R_Version(3, 5, 0)

#define HAS_ALTREP
#include <R_ext/Altrep.h>

// Returns the ALTREP class name or NULL if x is not an altrep
// object.
static inline const char* nanoarrow_altrep_class(SEXP x) {
  if (ALTREP(x)) {
    SEXP data_class_sym = CAR(ATTRIB(ALTREP_CLASS(x)));
    return CHAR(PRINTNAME(data_class_sym));
  } else {
    return NULL;
  }
}

#else

static inline const char* nanoarrow_altrep_class(SEXP x) { return NULL; }

#endif

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
