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

#ifndef R_NANOARROW_PRESERVE_H_INCLUDED
#define R_NANOARROW_PRESERVE_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Not really related to preserve/release, but needs C++
void intptr_as_string(intptr_t ptr_int, char* buf);

// Internal abstractions for R_PreserveObject and R_ReleaseObject
// that provide an opportunity for debugging information about
// preserved object lifecycle and possible future optimizations.
// These implementations use C++ and live in nanoarrow_cpp.cc
void nanoarrow_preserve_init(void);
void nanoarrow_preserve_sexp(SEXP obj);
void nanoarrow_release_sexp(SEXP obj);
int64_t nanoarrow_preserved_count(void);
int64_t nanoarrow_preserved_empty(void);
int nanoarrow_is_main_thread(void);

// For testing
void nanoarrow_preserve_and_release_on_other_thread(SEXP obj);

#ifdef __cplusplus
}
#endif

#endif
