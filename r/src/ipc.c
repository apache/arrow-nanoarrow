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

#include <stdint.h>
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

static void finalize_output_stream_xptr(SEXP output_stream_xptr) {
  struct ArrowIpcOutputStream* output_stream =
      (struct ArrowIpcOutputStream*)R_ExternalPtrAddr(output_stream_xptr);
  if (output_stream != NULL && output_stream->release != NULL) {
    output_stream->release(output_stream);
  }

  if (output_stream != NULL) {
    ArrowFree(output_stream);
  }
}

static SEXP output_stream_owning_xptr(void) {
  struct ArrowIpcOutputStream* output_stream =
      (struct ArrowIpcOutputStream*)ArrowMalloc(sizeof(struct ArrowIpcOutputStream));
  output_stream->release = NULL;
  SEXP output_stream_xptr =
      PROTECT(R_MakeExternalPtr(output_stream, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(output_stream_xptr, &finalize_output_stream_xptr);
  UNPROTECT(1);
  return output_stream_xptr;
}

static void finalize_writer_xptr(SEXP writer_xptr) {
  struct ArrowIpcWriter* writer = (struct ArrowIpcWriter*)R_ExternalPtrAddr(writer_xptr);
  if (writer != NULL && writer->private_data != NULL) {
    ArrowIpcWriterReset(writer);
  }

  if (writer != NULL) {
    ArrowFree(writer);
  }
}

static SEXP writer_owning_xptr(void) {
  struct ArrowIpcWriter* writer =
      (struct ArrowIpcWriter*)ArrowMalloc(sizeof(struct ArrowIpcWriter));
  writer->private_data = NULL;
  SEXP writer_xptr = PROTECT(R_MakeExternalPtr(writer, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(writer_xptr, &finalize_output_stream_xptr);
  UNPROTECT(1);
  return writer_xptr;
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

static SEXP handle_readbin_writebin_error(SEXP cond, void* hdata) {
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

static SEXP call_writebin(void* hdata) {
  struct ConnectionInputStreamHandler* data = (struct ConnectionInputStreamHandler*)hdata;

  // Write 1MB chunks
  int64_t chunk_buffer_size = 1048576;
  SEXP chunk_buffer = PROTECT(Rf_allocVector(RAWSXP, chunk_buffer_size));
  SEXP call = PROTECT(Rf_lang3(nanoarrow_sym_writebin, chunk_buffer, data->con));
  while (data->buf_size_bytes > chunk_buffer_size) {
    memcpy(RAW(chunk_buffer), data->buf, chunk_buffer_size);
    Rf_eval(call, R_BaseEnv);
    data->buf_size_bytes -= chunk_buffer_size;
    data->buf += chunk_buffer_size;
  }

  UNPROTECT(2);

  // Write remaining bytes
  if (data->buf_size_bytes > 0) {
    chunk_buffer = PROTECT(Rf_allocVector(RAWSXP, data->buf_size_bytes));
    call = PROTECT(Rf_lang3(nanoarrow_sym_writebin, chunk_buffer, data->con));
    memcpy(RAW(chunk_buffer), data->buf, data->buf_size_bytes);
    Rf_eval(call, R_BaseEnv);
    UNPROTECT(2);
  }

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

  R_tryCatchError(&call_readbin, &data, &handle_readbin_writebin_error, &data);
  return data.return_code;
}

static ArrowErrorCode write_con_output_stream(struct ArrowIpcOutputStream* stream,
                                              const void* buf, int64_t buf_size_bytes,
                                              int64_t* size_write_out,
                                              struct ArrowError* error) {
  if (!nanoarrow_is_main_thread()) {
    ArrowErrorSet(error, "Can't read from R connection on a non-R thread");
    return EIO;
  }

  struct ConnectionInputStreamHandler data;
  data.con = (SEXP)stream->private_data;
  data.buf = (void*)buf;
  data.buf_size_bytes = buf_size_bytes;
  data.size_read_out = NULL;
  data.error = error;
  data.return_code = NANOARROW_OK;

  R_tryCatchError(&call_writebin, &data, &handle_readbin_writebin_error, &data);

  // This implementation always blocks until all bytes have been written
  *size_write_out = buf_size_bytes;

  return data.return_code;
}

static void release_con_input_stream(struct ArrowIpcInputStream* stream) {
  nanoarrow_release_sexp((SEXP)stream->private_data);
}

static void release_con_output_stream(struct ArrowIpcOutputStream* stream) {
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

SEXP nanoarrow_c_ipc_output_stream_connection(SEXP con) {
  SEXP output_stream_xptr = PROTECT(output_stream_owning_xptr());
  struct ArrowIpcOutputStream* output_stream =
      (struct ArrowIpcOutputStream*)R_ExternalPtrAddr(output_stream_xptr);

  output_stream->write = &write_con_output_stream;
  output_stream->release = &release_con_output_stream;
  output_stream->private_data = (SEXP)con;
  nanoarrow_preserve_sexp(con);

  UNPROTECT(1);
  return output_stream_xptr;
}
