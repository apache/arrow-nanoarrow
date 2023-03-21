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

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "nanoarrow.h"
#include "nanoarrow_ipc.h"

void ArrowIpcInputStreamMove(struct ArrowIpcInputStream* src,
                             struct ArrowIpcInputStream* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcInputStream));
  src->release = NULL;
}

struct ArrowIpcInputStreamBufferPrivate {
  struct ArrowBuffer input;
  int64_t cursor_bytes;
};

static ArrowErrorCode ArrowIpcInputStreamBufferRead(struct ArrowIpcInputStream* stream,
                                                    void* buf, int64_t buf_size_bytes,
                                                    int64_t* size_read_out,
                                                    struct ArrowError* error) {
  if (buf_size_bytes == 0) {
    return NANOARROW_OK;
  }

  struct ArrowIpcInputStreamBufferPrivate* private_data =
      (struct ArrowIpcInputStreamBufferPrivate*)stream->private_data;
  int64_t bytes_remaining = private_data->input.size_bytes - private_data->cursor_bytes;
  int64_t bytes_to_read;
  if (bytes_remaining > buf_size_bytes) {
    bytes_to_read = buf_size_bytes;
  } else {
    bytes_to_read = bytes_remaining;
  }

  if (bytes_to_read > 0) {
    memcpy(buf, private_data->input.data + private_data->cursor_bytes, bytes_to_read);
  }

  *size_read_out = bytes_to_read;
  private_data->cursor_bytes += bytes_to_read;
  return NANOARROW_OK;
}

static void ArrowIpcInputStreamBufferRelease(struct ArrowIpcInputStream* stream) {
  struct ArrowIpcInputStreamBufferPrivate* private_data =
      (struct ArrowIpcInputStreamBufferPrivate*)stream->private_data;
  ArrowBufferReset(&private_data->input);
  ArrowFree(private_data);
  stream->release = NULL;
}

ArrowErrorCode ArrowIpcInputStreamInitBuffer(struct ArrowIpcInputStream* stream,
                                             struct ArrowBuffer* input) {
  struct ArrowIpcInputStreamBufferPrivate* private_data =
      (struct ArrowIpcInputStreamBufferPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcInputStreamBufferPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  ArrowBufferMove(input, &private_data->input);
  private_data->cursor_bytes = 0;
  stream->read = &ArrowIpcInputStreamBufferRead;
  stream->release = &ArrowIpcInputStreamBufferRelease;
  stream->private_data = private_data;

  return NANOARROW_OK;
}

struct ArrowIpcArrayStreamReaderPrivate {
  struct ArrowIpcInputStream input;
  struct ArrowIpcDecoder decoder;
  struct ArrowSchema out_schema;
  int64_t field_index;
  struct ArrowBuffer header;
  struct ArrowBuffer body;
  struct ArrowError error;
};

static void ArrowIpcArrayStreamReaderRelease(struct ArrowArrayStream* stream) {
  struct ArrowIpcArrayStreamReaderPrivate* private_data =
      (struct ArrowIpcArrayStreamReaderPrivate*)stream->private_data;

  if (private_data->input.release != NULL) {
    private_data->input.release(&private_data->input);
  }

  ArrowIpcDecoderReset(&private_data->decoder);

  if (private_data->out_schema.release != NULL) {
    private_data->out_schema.release(&private_data->out_schema);
  }

  ArrowBufferReset(&private_data->header);
  ArrowBufferReset(&private_data->body);

  ArrowFree(private_data);
  stream->release = NULL;
}

#define NANOARROW_IPC_ARRAY_STREAM_READER_CHUNK_SIZE 65536

static int ArrowIpcArrayStreamReaderRead(
    struct ArrowIpcArrayStreamReaderPrivate* private_data, struct ArrowBuffer* buffer,
    int64_t* bytes_read) {
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(
      &private_data->header, NANOARROW_IPC_ARRAY_STREAM_READER_CHUNK_SIZE));

  NANOARROW_RETURN_NOT_OK(private_data->input.read(
      &private_data->input, buffer->data + buffer->size_bytes,
      NANOARROW_IPC_ARRAY_STREAM_READER_CHUNK_SIZE, bytes_read, &private_data->error));

  buffer->size_bytes += *bytes_read;
  return NANOARROW_OK;
}

static int ArrowIpcArrayStreamReaderNextHeader(
    struct ArrowIpcArrayStreamReaderPrivate* private_data) {
  private_data->header.size_bytes = 0;
  struct ArrowBufferView input_view;

  int64_t bytes_read = 0;
  int result;
  do {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcArrayStreamReaderRead(private_data, &private_data->header, &bytes_read));
    input_view.data.data = private_data->header.data;
    input_view.size_bytes = private_data->header.size_bytes;
    result = ArrowIpcDecoderVerifyHeader(&private_data->decoder, input_view,
                                         &private_data->error);
  } while (result == ESPIPE || bytes_read == 0);

  if (result != NANOARROW_OK && bytes_read == 0) {
    return ENODATA;
  }

  NANOARROW_RETURN_NOT_OK(result);
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeHeader(&private_data->decoder, input_view,
                                                      &private_data->error));
  return NANOARROW_OK;
}

static int ArrowIpcArrayStreamReaderNextBody(
    struct ArrowIpcArrayStreamReaderPrivate* private_data) {
  int64_t bytes_read;
  int64_t bytes_to_read = private_data->decoder.body_size_bytes;

  // Reserve space in the body buffer
  private_data->body.size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&private_data->body, bytes_to_read));

  // Copy any body bytes from the header buffer
  int64_t extra_bytes_in_header =
      private_data->header.size_bytes - private_data->decoder.header_size_bytes;
  memcpy(
      private_data->body.data,
      private_data->header.data + private_data->header.size_bytes - extra_bytes_in_header,
      extra_bytes_in_header);

  // Read the rest of the body buffer
  NANOARROW_RETURN_NOT_OK(private_data->input.read(
      &private_data->input, private_data->body.data + extra_bytes_in_header,
      bytes_to_read - extra_bytes_in_header, &bytes_read, &private_data->error));

  // Set the size of the buffer
  private_data->body.size_bytes = bytes_to_read;

  return NANOARROW_OK;
}

static int ArrowIpcArrayStreamReaderReadSchemaIfNeeded(
    struct ArrowIpcArrayStreamReaderPrivate* private_data) {
  if (private_data->out_schema.release != NULL) {
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderNextHeader(private_data));

  // Error if this isn't a schema message
  if (private_data->decoder.message_type != NANOARROW_IPC_MESSAGE_TYPE_SCHEMA) {
    ArrowErrorSet(&private_data->error,
                  "Unexpected message type at start of input (expected Schema)");
    return EINVAL;
  }

  // ...or if it uses features we don't support
  if (private_data->decoder.feature_flags & NANOARROW_IPC_FEATURE_COMPRESSED_BODY) {
    ArrowErrorSet(&private_data->error,
                  "This stream uses unsupported feature COMPRESSED_BODY");
    return EINVAL;
  }

  if (private_data->decoder.feature_flags &
      NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT) {
    ArrowErrorSet(&private_data->error,
                  "This stream uses unsupported feature DICTIONARY_REPLACEMENT");
    return EINVAL;
  }

  // Notify the decoder of buffer endianness
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderSetEndianness(&private_data->decoder,
                                                       private_data->decoder.endianness));

  struct ArrowSchema tmp;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcDecoderDecodeSchema(&private_data->decoder, &tmp, &private_data->error));

  // Only support "read the whole thing" for now
  if (private_data->field_index != -1) {
    tmp.release(&tmp);
    ArrowErrorSet(&private_data->error, "Field index != -1 is not yet supported");
    return ENOTSUP;
  }

  // Notify the decoder of the schema for forthcoming messages
  int result =
      ArrowIpcDecoderSetSchema(&private_data->decoder, &tmp, &private_data->error);
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, &private_data->out_schema);
  return NANOARROW_OK;
}

static int ArrowIpcArrayStreamReaderGetSchema(struct ArrowArrayStream* stream,
                                              struct ArrowSchema* out) {
  struct ArrowIpcArrayStreamReaderPrivate* private_data =
      (struct ArrowIpcArrayStreamReaderPrivate*)stream->private_data;
  private_data->error.message[0] = '\0';
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderReadSchemaIfNeeded(private_data));
  return ArrowSchemaDeepCopy(&private_data->out_schema, out);
}

static int ArrowIpcArrayStreamReaderGetNext(struct ArrowArrayStream* stream,
                                            struct ArrowArray* out) {
  struct ArrowIpcArrayStreamReaderPrivate* private_data =
      (struct ArrowIpcArrayStreamReaderPrivate*)stream->private_data;
  // Check if we are all done
  if (private_data->input.release == NULL) {
    out->release = NULL;
    return NANOARROW_OK;
  }

  private_data->error.message[0] = '\0';
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderReadSchemaIfNeeded(private_data));

  // Read + decode the next header
  int result = ArrowIpcArrayStreamReaderNextHeader(private_data);
  if (result == ENODATA) {
    // If the stream is finished, release the input
    private_data->input.release(&private_data->input);
    out->release = NULL;
    return NANOARROW_OK;
  }

  // Make sure we have a RecordBatch message
  if (private_data->decoder.message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH) {
    ArrowErrorSet(&private_data->error, "Unexpected message type (expected RecordBatch)");
    return EINVAL;
  }

  // Read in the body
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderNextBody(private_data));

  struct ArrowBufferView body_view;
  body_view.data.data = private_data->body.data + private_data->body.size_bytes;
  body_view.size_bytes = 0;

  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeArray(&private_data->decoder, body_view,
                                                     private_data->field_index, out,
                                                     &private_data->error));

  return NANOARROW_OK;
}

static const char* ArrowIpcArrayStreamReaderGetLastError(
    struct ArrowArrayStream* stream) {
  struct ArrowIpcArrayStreamReaderPrivate* private_data =
      (struct ArrowIpcArrayStreamReaderPrivate*)stream->private_data;
  return private_data->error.message;
}

ArrowErrorCode ArrowIpcArrayStreamReaderInit(
    struct ArrowArrayStream* out, struct ArrowIpcInputStream* input_stream,
    struct ArrowIpcArrayStreamReaderOptions options) {
  struct ArrowIpcArrayStreamReaderPrivate* private_data =
      (struct ArrowIpcArrayStreamReaderPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcArrayStreamReaderPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  int result = ArrowIpcDecoderInit(&private_data->decoder);
  if (result != NANOARROW_OK) {
    ArrowFree(private_data);
    return result;
  }

  ArrowBufferInit(&private_data->header);
  ArrowBufferInit(&private_data->body);
  private_data->out_schema.release = NULL;
  ArrowIpcInputStreamMove(input_stream, &private_data->input);

  out->private_data = private_data;
  out->get_schema = &ArrowIpcArrayStreamReaderGetSchema;
  out->get_next = &ArrowIpcArrayStreamReaderGetNext;
  out->get_last_error = &ArrowIpcArrayStreamReaderGetLastError;
  out->release = &ArrowIpcArrayStreamReaderRelease;

  return NANOARROW_OK;
}
