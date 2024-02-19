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

#include "buffer.h"
#include "nanoarrow/r.h"
#include "nanoarrow_ipc.h"

static void finalize_input_stream_xptr(SEXP input_stream_xptr) {
  struct ArrowIpcInputStream* input_stream =
      (struct ArrowIpcInputStream*)R_ExternalPtrAddr(input_stream_xptr);
  if (input_stream != NULL && input_stream->release != NULL) {
    input_stream->release(input_stream);
  }

  if (input_stream != NULL) {
    ArrowFree(input_stream);
  }
}

static SEXP input_stream_owning_xptr(void) {
  struct ArrowIpcInputStream* input_stream =
      (struct ArrowIpcInputStream*)ArrowMalloc(sizeof(struct ArrowIpcInputStream));
  input_stream->release = NULL;
  SEXP input_stream_xptr =
      PROTECT(R_MakeExternalPtr(input_stream, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(input_stream_xptr, &finalize_input_stream_xptr);
  UNPROTECT(1);
  return input_stream_xptr;
}

SEXP nanoarrow_c_ipc_array_reader_buffer(SEXP buffer_xptr) {
  struct ArrowBuffer* buffer = buffer_from_xptr(buffer_xptr);

  SEXP array_stream_xptr = PROTECT(nanoarrow_array_stream_owning_xptr());
  struct ArrowArrayStream* array_stream =
      nanoarrow_output_array_stream_from_xptr(array_stream_xptr);

  SEXP input_stream_xptr = PROTECT(input_stream_owning_xptr());
  struct ArrowIpcInputStream* input_stream =
      (struct ArrowIpcInputStream*)R_ExternalPtrAddr(input_stream_xptr);

  int code = ArrowIpcInputStreamInitBuffer(input_stream, buffer);
  if (code != NANOARROW_OK) {
    Rf_error("ArrowIpcInputStreamInitBuffer() failed");
  }

  code = ArrowIpcArrayStreamReaderInit(array_stream, input_stream, NULL);
  if (code != NANOARROW_OK) {
    Rf_error("ArrowIpcArrayStreamReaderInit() failed");
  }

  UNPROTECT(2);
  return array_stream_xptr;
}
