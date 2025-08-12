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

SEXP nanoarrow_c_version(void) {
  Rf_error("%ld", __STDC_VERSION__);
  return Rf_mkString(NANOARROW_VERSION);
}

SEXP nanoarrow_c_version_runtime(void) { return Rf_mkString(ArrowNanoarrowVersion()); }

SEXP nanoarrow_c_with_zstd(void) {
#if defined(NANOARROW_IPC_WITH_ZSTD)
  return Rf_ScalarLogical(1);
#else
  return Rf_ScalarLogical(0);
#endif
}
