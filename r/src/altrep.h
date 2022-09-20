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

// ALTREP available in R >= 3.5
#if defined(R_VERSION) && R_VERSION >= R_Version(3, 5, 0)
#define HAS_ALTREP
#include <R_ext/Altrep.h>
#endif


// ...ALTREP raw() class available in R >= 3.6
#if (defined(R_VERSION) && R_VERSION >= R_Version(3, 6, 0))
#define HAS_ALTREP_RAW
#endif

void register_nanoarrow_altrep(DllInfo* info);

#endif
