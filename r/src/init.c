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

/* generated by tools/make-callentries.R */
extern SEXP nanoarrow_c_array_stream_get_schema(SEXP array_stream_xptr);
extern SEXP nanoarrow_c_array_stream_get_next(SEXP array_stream_xptr);
extern SEXP nanoarrow_c_array_set_schema(SEXP array_xptr, SEXP schema_xptr, SEXP validate_sexp);
extern SEXP nanoarrow_c_infer_schema_array(SEXP array_xptr);
extern SEXP nanoarrow_c_array_view(SEXP array_xptr, SEXP schema_xptr);
extern SEXP nanoarrow_c_array_info(SEXP array_xptr, SEXP array_view_xptr, SEXP recursive_sexp);
extern SEXP nanoarrow_c_buffer_info(SEXP buffer_xptr);
extern SEXP nanoarrow_c_buffer_as_raw(SEXP buffer_xptr);
extern SEXP nanoarrow_c_build_id();
extern SEXP nanoarrow_c_build_id_runtime();
extern SEXP nanoarrow_c_allocate_schema();
extern SEXP nanoarrow_c_allocate_array();
extern SEXP nanoarrow_c_allocate_array_stream();
extern SEXP nanoarrow_c_pointer(SEXP obj_sexp);
extern SEXP nanoarrow_c_pointer_addr_dbl(SEXP ptr);
extern SEXP nanoarrow_c_pointer_addr_chr(SEXP ptr);
extern SEXP nanoarrow_c_pointer_addr_pretty(SEXP ptr);
extern SEXP nanoarrow_c_pointer_is_valid(SEXP ptr);
extern SEXP nanoarrow_c_pointer_release(SEXP ptr);
extern SEXP nanoarrow_c_pointer_move(SEXP ptr_src, SEXP ptr_dst);
extern SEXP nanoarrow_c_export_schema(SEXP schema_xptr, SEXP ptr_dst);
extern SEXP nanoarrow_c_export_array(SEXP array_xptr, SEXP ptr_dst);
extern SEXP nanoarrow_c_schema_to_list(SEXP schema_xptr);

static const R_CallMethodDef CallEntries[] = {
    {"nanoarrow_c_array_stream_get_schema", (DL_FUNC)&nanoarrow_c_array_stream_get_schema, 1},
    {"nanoarrow_c_array_stream_get_next", (DL_FUNC)&nanoarrow_c_array_stream_get_next, 1},
    {"nanoarrow_c_array_set_schema", (DL_FUNC)&nanoarrow_c_array_set_schema, 3},
    {"nanoarrow_c_infer_schema_array", (DL_FUNC)&nanoarrow_c_infer_schema_array, 1},
    {"nanoarrow_c_array_view", (DL_FUNC)&nanoarrow_c_array_view, 2},
    {"nanoarrow_c_array_info", (DL_FUNC)&nanoarrow_c_array_info, 3},
    {"nanoarrow_c_buffer_info", (DL_FUNC)&nanoarrow_c_buffer_info, 1},
    {"nanoarrow_c_buffer_as_raw", (DL_FUNC)&nanoarrow_c_buffer_as_raw, 1},
    {"nanoarrow_c_build_id", (DL_FUNC)&nanoarrow_c_build_id, 0},
    {"nanoarrow_c_build_id_runtime", (DL_FUNC)&nanoarrow_c_build_id_runtime, 0},
    {"nanoarrow_c_allocate_schema", (DL_FUNC)&nanoarrow_c_allocate_schema, 0},
    {"nanoarrow_c_allocate_array", (DL_FUNC)&nanoarrow_c_allocate_array, 0},
    {"nanoarrow_c_allocate_array_stream", (DL_FUNC)&nanoarrow_c_allocate_array_stream, 0},
    {"nanoarrow_c_pointer", (DL_FUNC)&nanoarrow_c_pointer, 1},
    {"nanoarrow_c_pointer_addr_dbl", (DL_FUNC)&nanoarrow_c_pointer_addr_dbl, 1},
    {"nanoarrow_c_pointer_addr_chr", (DL_FUNC)&nanoarrow_c_pointer_addr_chr, 1},
    {"nanoarrow_c_pointer_addr_pretty", (DL_FUNC)&nanoarrow_c_pointer_addr_pretty, 1},
    {"nanoarrow_c_pointer_is_valid", (DL_FUNC)&nanoarrow_c_pointer_is_valid, 1},
    {"nanoarrow_c_pointer_release", (DL_FUNC)&nanoarrow_c_pointer_release, 1},
    {"nanoarrow_c_pointer_move", (DL_FUNC)&nanoarrow_c_pointer_move, 2},
    {"nanoarrow_c_export_schema", (DL_FUNC)&nanoarrow_c_export_schema, 2},
    {"nanoarrow_c_export_array", (DL_FUNC)&nanoarrow_c_export_array, 2},
    {"nanoarrow_c_schema_to_list", (DL_FUNC)&nanoarrow_c_schema_to_list, 1},
    {NULL, NULL, 0}};
/* end generated by tools/make-callentries.R */

void R_init_nanoarrow(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
