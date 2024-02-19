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

#include "nanoarrow_ipc.h"

#include "buffer.h"
#include "nanoarrow/r.h"
#include "util.h"

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

struct ConnectionInputStreamHandler {
  SEXP con;
  uint8_t* buf;
  int64_t buf_size_bytes;
  int64_t* size_read_out;
  struct ArrowError* error;
  int return_code;
};

static SEXP handle_readbin_error(SEXP cond, void* hdata) {
  struct ConnectionInputStreamHandler* data = (struct ConnectionInputStreamHandler*)hdata;

  SEXP fun = PROTECT(Rf_install("conditionMessage"));
  SEXP call = PROTECT(Rf_lang2(fun, cond));
  SEXP result = PROTECT(Rf_eval(call, R_BaseEnv));
  SEXP result0 = STRING_ELT(result, 0);
  const char* cond_msg = Rf_translateCharUTF8(result0);

  ArrowErrorSet(data->error, "R execution error: %s", cond_msg);
  data->return_code = EIO;

  UNPROTECT(3);
  return R_NilValue;
}

static SEXP call_readbin(void* hdata) {
  struct ConnectionInputStreamHandler* data = (struct ConnectionInputStreamHandler*)hdata;
  SEXP n = PROTECT(Rf_ScalarReal((double)data->buf_size_bytes));
  SEXP call = PROTECT(Rf_lang4(nanoarrow_sym_readbin, data->con, nanoarrow_ptype_raw, n));

  SEXP result = PROTECT(Rf_eval(call, R_BaseEnv));
  R_xlen_t bytes_read = Rf_xlength(result);
  memcpy(data->buf, RAW(result), bytes_read);
  *(data->size_read_out) = bytes_read;

  UNPROTECT(3);
  return R_NilValue;
}

static ArrowErrorCode read_con_input_stream(struct ArrowIpcInputStream* stream,
                                            uint8_t* buf, int64_t buf_size_bytes,
                                            int64_t* size_read_out,
                                            struct ArrowError* error) {
  if (!nanoarrow_is_main_thread()) {
    ArrowErrorSet(error, "Can't read from R connection on a non-R thread");
    return EIO;
  }

  struct ConnectionInputStreamHandler data;
  data.con = (SEXP)stream->private_data;
  data.buf = buf;
  data.buf_size_bytes = buf_size_bytes;
  data.size_read_out = size_read_out;
  data.error = error;
  data.return_code = NANOARROW_OK;

  R_tryCatchError(&call_readbin, &data, &handle_readbin_error, &data);
  return data.return_code;
}

static void release_con_input_stream(struct ArrowIpcInputStream* stream) {
  nanoarrow_release_sexp((SEXP)stream->private_data);
}

SEXP nanoarrow_c_ipc_array_reader_connection(SEXP con) {
  SEXP array_stream_xptr = PROTECT(nanoarrow_array_stream_owning_xptr());
  struct ArrowArrayStream* array_stream =
      nanoarrow_output_array_stream_from_xptr(array_stream_xptr);

  SEXP input_stream_xptr = PROTECT(input_stream_owning_xptr());
  struct ArrowIpcInputStream* input_stream =
      (struct ArrowIpcInputStream*)R_ExternalPtrAddr(input_stream_xptr);

  input_stream->read = &read_con_input_stream;
  input_stream->release = &release_con_input_stream;
  input_stream->private_data = (SEXP)con;
  nanoarrow_preserve_sexp(con);

  int code = ArrowIpcArrayStreamReaderInit(array_stream, input_stream, NULL);
  if (code != NANOARROW_OK) {
    Rf_error("ArrowIpcArrayStreamReaderInit() failed");
  }

  UNPROTECT(2);
  return array_stream_xptr;
}
