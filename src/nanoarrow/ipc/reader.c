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
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

// R 3.6 / Windows builds on a very old toolchain that does not define ENODATA
#if defined(_WIN32) && !defined(_MSC_VER) && !defined(ENODATA)
#define ENODATA 120
#endif

// Sentinel value to indicate that we haven't read a message yet
// and don't know the number of header prefix bytes to expect.
static const int32_t kExpectedHeaderPrefixSizeNotSet = -1;

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
                                                    uint8_t* buf, int64_t buf_size_bytes,
                                                    int64_t* size_read_out,
                                                    struct ArrowError* error) {
  NANOARROW_UNUSED(error);

  if (buf_size_bytes == 0) {
    *size_read_out = 0;
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
  NANOARROW_DCHECK(stream != NULL);

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

struct ArrowIpcInputStreamFilePrivate {
  FILE* file_ptr;
  int stream_finished;
  int close_on_release;
};

static void ArrowIpcInputStreamFileRelease(struct ArrowIpcInputStream* stream) {
  struct ArrowIpcInputStreamFilePrivate* private_data =
      (struct ArrowIpcInputStreamFilePrivate*)stream->private_data;

  if (private_data->file_ptr != NULL && private_data->close_on_release) {
    fclose(private_data->file_ptr);
  }

  ArrowFree(private_data);
  stream->release = NULL;
}

static ArrowErrorCode ArrowIpcInputStreamFileRead(struct ArrowIpcInputStream* stream,
                                                  uint8_t* buf, int64_t buf_size_bytes,
                                                  int64_t* size_read_out,
                                                  struct ArrowError* error) {
  struct ArrowIpcInputStreamFilePrivate* private_data =
      (struct ArrowIpcInputStreamFilePrivate*)stream->private_data;

  if (private_data->stream_finished) {
    *size_read_out = 0;
    return NANOARROW_OK;
  }

  // Do the read
  int64_t bytes_read = (int64_t)fread(buf, 1, buf_size_bytes, private_data->file_ptr);
  *size_read_out = bytes_read;

  if (bytes_read != buf_size_bytes) {
    private_data->stream_finished = 1;

    // Inspect error
    int has_error = !feof(private_data->file_ptr) && ferror(private_data->file_ptr);

    // Try to close the file now
    if (private_data->close_on_release) {
      if (fclose(private_data->file_ptr) == 0) {
        private_data->file_ptr = NULL;
      }
    }

    // Maybe return error
    if (has_error) {
      ArrowErrorSet(error, "ArrowIpcInputStreamFile IO error");
      return EIO;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcInputStreamInitFile(struct ArrowIpcInputStream* stream,
                                           void* file_ptr, int close_on_release) {
  NANOARROW_DCHECK(stream != NULL);
  if (file_ptr == NULL) {
    return errno ? errno : EINVAL;
  }

  struct ArrowIpcInputStreamFilePrivate* private_data =
      (struct ArrowIpcInputStreamFilePrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcInputStreamFilePrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->file_ptr = (FILE*)file_ptr;
  private_data->close_on_release = close_on_release;
  private_data->stream_finished = 0;

  stream->read = &ArrowIpcInputStreamFileRead;
  stream->release = &ArrowIpcInputStreamFileRelease;
  stream->private_data = private_data;
  return NANOARROW_OK;
}

struct ArrowIpcArrayStreamReaderPrivate {
  struct ArrowIpcInputStream input;
  struct ArrowIpcDecoder decoder;
  int use_shared_buffers;
  struct ArrowSchema out_schema;
  int64_t field_index;
  struct ArrowBuffer header;
  struct ArrowBuffer body;
  int32_t expected_header_prefix_size;
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
    ArrowSchemaRelease(&private_data->out_schema);
  }

  ArrowBufferReset(&private_data->header);
  ArrowBufferReset(&private_data->body);

  ArrowFree(private_data);
  stream->release = NULL;
}

static int ArrowIpcArrayStreamReaderNextHeader(
    struct ArrowIpcArrayStreamReaderPrivate* private_data,
    enum ArrowIpcMessageType message_type) {
  private_data->header.size_bytes = 0;
  int64_t bytes_read = 0;

  // Read 8 bytes (continuation + header size in bytes)
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(&private_data->header, 8),
                                     &private_data->error);
  NANOARROW_RETURN_NOT_OK(private_data->input.read(&private_data->input,
                                                   private_data->header.data, 8,
                                                   &bytes_read, &private_data->error));
  private_data->header.size_bytes += bytes_read;

  if (bytes_read == 0) {
    // The caller might not use this error message (e.g., if the end of the stream
    // is one of the valid outcomes) but we set the error anyway in case it gets
    // propagated higher (e.g., if the stream is empty and there's no schema message)
    ArrowErrorSet(&private_data->error, "No data available on stream");
    return ENODATA;
  } else if (bytes_read == 4 && private_data->expected_header_prefix_size == 4) {
    // Special case very, very old IPC streams that used 0x00000000 as the
    // end-of-stream indicator. We may want to remove this case at some point:
    // https://github.com/apache/arrow-nanoarrow/issues/648
    uint32_t last_four_bytes = 0;
    memcpy(&last_four_bytes, private_data->header.data, sizeof(uint32_t));
    if (last_four_bytes == 0) {
      ArrowErrorSet(&private_data->error, "No data available on stream");
      return ENODATA;
    } else {
      ArrowErrorSet(&private_data->error,
                    "Expected 0x00000000 if exactly four bytes are available at the end "
                    "of a stream");
      return EINVAL;
    }
  } else if (bytes_read != 8) {
    ArrowErrorSet(&private_data->error,
                  "Expected at least 8 bytes in remainder of stream");
    return EINVAL;
  }

  struct ArrowBufferView input_view;
  input_view.data.data = private_data->header.data;
  input_view.size_bytes = private_data->header.size_bytes;

  // Use PeekHeader to fill in decoder.header_size_bytes
  int32_t prefix_size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderPeekHeader(
      &private_data->decoder, input_view, &prefix_size_bytes, &private_data->error));

  // Check for a consistent header prefix size
  if (private_data->expected_header_prefix_size != kExpectedHeaderPrefixSizeNotSet &&
      prefix_size_bytes != private_data->expected_header_prefix_size) {
    ArrowErrorSet(&private_data->error,
                  "Expected prefix %d prefix header bytes but found %d",
                  (int)private_data->expected_header_prefix_size, (int)prefix_size_bytes);
    return EINVAL;
  } else {
    private_data->expected_header_prefix_size = prefix_size_bytes;
  }

  // Legacy streams are missing the 0xFFFFFFFF at the start of the message. The
  // decoder can handle this; however, verification will fail because flatbuffers
  // must be 8-byte aligned. To handle this case, we prepend the continuation
  // token to the start of the stream and ensure that we read four fewer bytes
  // the next time we issue a read. We may be able to remove this case in the future:
  // https://github.com/apache/arrow-nanoarrow/issues/648
  int64_t extra_bytes_already_read = 0;
  if (prefix_size_bytes == 4) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(&private_data->header, 4),
                                       &private_data->error);
    memmove(private_data->header.data + 4, private_data->header.data,
            private_data->header.size_bytes);
    uint32_t continuation = 0xFFFFFFFFU;
    memcpy(private_data->header.data, &continuation, sizeof(uint32_t));
    private_data->header.size_bytes += 4;
    extra_bytes_already_read = 4;

    input_view.data.data = private_data->header.data;
    input_view.size_bytes = private_data->header.size_bytes;

    int32_t new_prefix_size_bytes;
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderPeekHeader(&private_data->decoder, input_view,
                                                      &new_prefix_size_bytes,
                                                      &private_data->error));
    NANOARROW_DCHECK(new_prefix_size_bytes == 8);
  }

  // Read the header bytes
  int64_t expected_header_bytes = private_data->decoder.header_size_bytes - 8;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferReserve(&private_data->header,
                         expected_header_bytes - extra_bytes_already_read),
      &private_data->error);
  NANOARROW_RETURN_NOT_OK(private_data->input.read(
      &private_data->input, private_data->header.data + private_data->header.size_bytes,
      expected_header_bytes - extra_bytes_already_read, &bytes_read,
      &private_data->error));
  private_data->header.size_bytes += bytes_read;

  // Verify + decode the header
  input_view.data.data = private_data->header.data;
  input_view.size_bytes = private_data->header.size_bytes;
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderVerifyHeader(&private_data->decoder, input_view,
                                                      &private_data->error));

  // If we have a 4-byte header prefix, make sure the metadata version is V4
  // (Note that some V4 IPC files have an 8 byte header prefix).
  if (prefix_size_bytes == 4 &&
      private_data->decoder.metadata_version != NANOARROW_IPC_METADATA_VERSION_V4) {
    ArrowErrorSet(&private_data->error,
                  "Header prefix size of four bytes is only allowed for V4 metadata");
    return EINVAL;
  }

  // Don't decode the message if it's of the wrong type (because the error message
  // is better communicated by the caller)
  if (private_data->decoder.message_type != message_type) {
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeHeader(&private_data->decoder, input_view,
                                                      &private_data->error));
  return NANOARROW_OK;
}

static int ArrowIpcArrayStreamReaderNextBody(
    struct ArrowIpcArrayStreamReaderPrivate* private_data) {
  int64_t bytes_read;
  int64_t bytes_to_read = private_data->decoder.body_size_bytes;

  // Read the body bytes
  private_data->body.size_bytes = 0;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferReserve(&private_data->body, bytes_to_read), &private_data->error);
  NANOARROW_RETURN_NOT_OK(private_data->input.read(&private_data->input,
                                                   private_data->body.data, bytes_to_read,
                                                   &bytes_read, &private_data->error));
  private_data->body.size_bytes += bytes_read;

  if (bytes_read != bytes_to_read) {
    ArrowErrorSet(&private_data->error,
                  "Expected to be able to read %" PRId64
                  " bytes for message body but got %" PRId64,
                  bytes_to_read, bytes_read);
    return ESPIPE;
  } else {
    return NANOARROW_OK;
  }
}

static int ArrowIpcArrayStreamReaderReadSchemaIfNeeded(
    struct ArrowIpcArrayStreamReaderPrivate* private_data) {
  if (private_data->out_schema.release != NULL) {
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderNextHeader(
      private_data, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA));

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
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcDecoderSetEndianness(&private_data->decoder,
                                   private_data->decoder.endianness),
      &private_data->error);

  struct ArrowSchema tmp;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcDecoderDecodeSchema(&private_data->decoder, &tmp, &private_data->error));

  // Only support "read the whole thing" for now
  if (private_data->field_index != -1) {
    ArrowSchemaRelease(&tmp);
    ArrowErrorSet(&private_data->error, "Field index != -1 is not yet supported");
    return ENOTSUP;
  }

  // Notify the decoder of the schema for forthcoming messages
  int result =
      ArrowIpcDecoderSetSchema(&private_data->decoder, &tmp, &private_data->error);
  if (result != NANOARROW_OK) {
    ArrowSchemaRelease(&tmp);
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
  ArrowErrorInit(&private_data->error);
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderReadSchemaIfNeeded(private_data));

  // Read + decode the next header
  int result = ArrowIpcArrayStreamReaderNextHeader(
      private_data, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);
  if (result == ENODATA) {
    // Stream is finished either because there is no input or because
    // end of stream bytes were read.
    out->release = NULL;
    return NANOARROW_OK;
  } else if (result != NANOARROW_OK) {
    // Other error
    return result;
  }

  // Make sure we have a RecordBatch message
  switch (private_data->decoder.message_type) {
    case NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH:
      break;
    case NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH:
      ArrowErrorSet(
          &private_data->error,
          "Found valid dictionary batch but dictionary encoding is not yet supported");
      return ENOTSUP;
    default:
      ArrowErrorSet(&private_data->error,
                    "Unexpected message type (expected RecordBatch)");
      return EINVAL;
  }

  // Read in the body
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamReaderNextBody(private_data));

  struct ArrowArray tmp;

  if (private_data->use_shared_buffers) {
    struct ArrowIpcSharedBuffer shared;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcSharedBufferInit(&shared, &private_data->body), &private_data->error);
    result = ArrowIpcDecoderDecodeArrayFromShared(
        &private_data->decoder, &shared, private_data->field_index, &tmp,
        NANOARROW_VALIDATION_LEVEL_FULL, &private_data->error);
    ArrowIpcSharedBufferReset(&shared);
    NANOARROW_RETURN_NOT_OK(result);
  } else {
    struct ArrowBufferView body_view;
    body_view.data.data = private_data->body.data;
    body_view.size_bytes = private_data->body.size_bytes;

    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeArray(
        &private_data->decoder, body_view, private_data->field_index, &tmp,
        NANOARROW_VALIDATION_LEVEL_FULL, &private_data->error));
  }

  ArrowArrayMove(&tmp, out);
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
    struct ArrowIpcArrayStreamReaderOptions* options) {
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
  private_data->expected_header_prefix_size = kExpectedHeaderPrefixSizeNotSet;

  if (options != NULL) {
    private_data->field_index = options->field_index;
    private_data->use_shared_buffers = options->use_shared_buffers;
  } else {
    private_data->field_index = -1;
    private_data->use_shared_buffers = ArrowIpcSharedBufferIsThreadSafe();
  }

  out->private_data = private_data;
  out->get_schema = &ArrowIpcArrayStreamReaderGetSchema;
  out->get_next = &ArrowIpcArrayStreamReaderGetNext;
  out->get_last_error = &ArrowIpcArrayStreamReaderGetLastError;
  out->release = &ArrowIpcArrayStreamReaderRelease;

  return NANOARROW_OK;
}
