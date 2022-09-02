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

#include "nanoarrow/nanoarrow.h"

#include "library.h"

const char* my_library_nanoarrow_build_id_runtime() { return ArrowNanoarrowBuildId(); }

const char* my_library_nanoarrow_build_id_compile_time() { return NANOARROW_BUILD_ID; }

// TODO: when namespacing PR is merged, make sure this works
#define STR(x) #x
const char* my_library_nanoarrow_namespace() { return STR(NANOARROW_NAMESPACE); }
