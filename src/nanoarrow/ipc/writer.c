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

void ArrowIpcOutputStreamMove(struct ArrowIpcOutputStream* src,
                              struct ArrowIpcOutputStream* dst) {
  NANOARROW_DCHECK(src != NULL && dst != NULL);

  memcpy(dst, src, sizeof(struct ArrowIpcOutputStream));
  src->release = NULL;
}

struct ArrowIpcOutputStreamBufferPrivate {
  struct ArrowBuffer* output;
};

static ArrowErrorCode ArrowIpcOutputStreamBufferWrite(struct ArrowIpcOutputStream* stream,
                                                      const void* buf,
                                                      int64_t buf_size_bytes,
                                                      int64_t* size_written_out,
                                                      struct ArrowError* error) {
  struct ArrowIpcOutputStreamBufferPrivate* private_data =
      (struct ArrowIpcOutputStreamBufferPrivate*)stream->private_data;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppend(private_data->output, buf, buf_size_bytes), error);
  *size_written_out = buf_size_bytes;
  return NANOARROW_OK;
}

static void ArrowIpcOutputStreamBufferRelease(struct ArrowIpcOutputStream* stream) {
  struct ArrowIpcOutputStreamBufferPrivate* private_data =
      (struct ArrowIpcOutputStreamBufferPrivate*)stream->private_data;
  ArrowFree(private_data);
  stream->release = NULL;
}

ArrowErrorCode ArrowIpcOutputStreamInitBuffer(struct ArrowIpcOutputStream* stream,
                                              struct ArrowBuffer* output) {
  NANOARROW_DCHECK(stream != NULL && output != NULL);

  struct ArrowIpcOutputStreamBufferPrivate* private_data =
      (struct ArrowIpcOutputStreamBufferPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcOutputStreamBufferPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->output = output;
  stream->write = &ArrowIpcOutputStreamBufferWrite;
  stream->release = &ArrowIpcOutputStreamBufferRelease;
  stream->private_data = private_data;

  return NANOARROW_OK;
}

struct ArrowIpcOutputStreamFilePrivate {
  FILE* file_ptr;
  int stream_finished;
  int close_on_release;
};

static void ArrowIpcOutputStreamFileRelease(struct ArrowIpcOutputStream* stream) {
  struct ArrowIpcOutputStreamFilePrivate* private_data =
      (struct ArrowIpcOutputStreamFilePrivate*)stream->private_data;

  if (private_data->file_ptr != NULL && private_data->close_on_release) {
    fclose(private_data->file_ptr);
  }

  ArrowFree(private_data);
  stream->release = NULL;
}

static ArrowErrorCode ArrowIpcOutputStreamFileWrite(struct ArrowIpcOutputStream* stream,
                                                    const void* buf,
                                                    int64_t buf_size_bytes,
                                                    int64_t* size_written_out,
                                                    struct ArrowError* error) {
  struct ArrowIpcOutputStreamFilePrivate* private_data =
      (struct ArrowIpcOutputStreamFilePrivate*)stream->private_data;

  if (private_data->stream_finished) {
    *size_written_out = 0;
    return NANOARROW_OK;
  }

  // Do the write
  int64_t bytes_written = (int64_t)fwrite(buf, 1, buf_size_bytes, private_data->file_ptr);
  *size_written_out = bytes_written;

  if (bytes_written != buf_size_bytes) {
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
      ArrowErrorSet(error, "ArrowIpcOutputStreamFile IO error");
      return EIO;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcOutputStreamInitFile(struct ArrowIpcOutputStream* stream,
                                            void* file_ptr, int close_on_release) {
  NANOARROW_DCHECK(stream != NULL);
  if (file_ptr == NULL) {
    return EINVAL;
  }

  struct ArrowIpcOutputStreamFilePrivate* private_data =
      (struct ArrowIpcOutputStreamFilePrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcOutputStreamFilePrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->file_ptr = (FILE*)file_ptr;
  private_data->close_on_release = close_on_release;
  private_data->stream_finished = 0;

  stream->write = &ArrowIpcOutputStreamFileWrite;
  stream->release = &ArrowIpcOutputStreamFileRelease;
  stream->private_data = private_data;
  return NANOARROW_OK;
}

struct ArrowIpcArrayStreamWriterPrivate {
  struct ArrowArrayStream in;
  struct ArrowIpcOutputStream output_stream;
  struct ArrowIpcEncoder encoder;
  struct ArrowSchema schema;
  struct ArrowArray array;
  struct ArrowArrayView array_view;
  struct ArrowBuffer buffer;
  struct ArrowBuffer body_buffer;
  int64_t buffer_cursor;
  int64_t body_buffer_cursor;
};

ArrowErrorCode ArrowIpcArrayStreamWriterInit(struct ArrowIpcArrayStreamWriter* writer,
                                             struct ArrowArrayStream* in,
                                             struct ArrowIpcOutputStream* output_stream) {
  NANOARROW_DCHECK(writer != NULL && in != NULL && output_stream != NULL);

  struct ArrowIpcArrayStreamWriterPrivate* private =
      (struct ArrowIpcArrayStreamWriterPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcArrayStreamWriterPrivate));

  if (private == NULL) {
    return ENOMEM;
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderInit(&private->encoder));
  ArrowIpcOutputStreamMove(output_stream, &private->output_stream);
  ArrowArrayStreamMove(in, &private->in);
  private->schema.release = NULL;
  private->array.release = NULL;
  ArrowArrayViewInitFromType(&private->array_view, NANOARROW_TYPE_UNINITIALIZED);
  ArrowBufferInit(&private->buffer);
  ArrowBufferInit(&private->body_buffer);
  private->buffer_cursor = 0;
  private->body_buffer_cursor = 0;

  writer->finished = 0;
  writer->private_data = private;
  return NANOARROW_OK;
}

void ArrowIpcArrayStreamWriterReset(struct ArrowIpcArrayStreamWriter* writer) {
  NANOARROW_DCHECK(writer != NULL);

  struct ArrowIpcArrayStreamWriterPrivate* private =
      (struct ArrowIpcArrayStreamWriterPrivate*)writer->private_data;

  if (private != NULL) {
    ArrowIpcEncoderReset(&private->encoder);
    ArrowArrayStreamRelease(&private->in);
    private->output_stream.release(&private->output_stream);
    if (private->schema.release != NULL) {
      ArrowSchemaRelease(&private->schema);
    }
    if (private->array.release != NULL) {
      ArrowArrayRelease(&private->array);
    }
    ArrowArrayViewReset(&private->array_view);
    ArrowBufferReset(&private->buffer);
    ArrowBufferReset(&private->body_buffer);

    ArrowFree(private);
  }
  memset(writer, 0, sizeof(struct ArrowIpcArrayStreamWriter));
}

static ArrowErrorCode ArrowIpcArrayStreamWriterPush(
    struct ArrowIpcArrayStreamWriterPrivate* private, struct ArrowBuffer* buffer,
    int* had_bytes_to_push, struct ArrowError* error) {
  int64_t* cursor = buffer == &private->buffer  //
                        ? &private->buffer_cursor
                        : &private->body_buffer_cursor;

  *had_bytes_to_push = *cursor < buffer->size_bytes;
  if (*had_bytes_to_push) {
    // bytes remain in the buffer; push those
    int64_t bytes_written;
    NANOARROW_RETURN_NOT_OK(private->output_stream.write(
        &private->output_stream, buffer->data + *cursor, buffer->size_bytes - *cursor,
        &bytes_written, error));
    *cursor += bytes_written;

    if (*cursor == buffer->size_bytes) {
      *cursor = buffer->size_bytes = 0;
    }
  }
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcArrayStreamWriterWriteSome(
    struct ArrowIpcArrayStreamWriter* writer, struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL);

  if (writer->finished) {
    ArrowErrorSet(error, "ArrowIpcArrayStreamWriterWriteSome on a finished writer");
    return EINVAL;
  }

  struct ArrowIpcArrayStreamWriterPrivate* private =
      (struct ArrowIpcArrayStreamWriterPrivate*)writer->private_data;

  int had_bytes_to_push = 0;

  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamWriterPush(private, &private->buffer,
                                                        &had_bytes_to_push, error));
  if (had_bytes_to_push) {
    return NANOARROW_OK;
  }
  // buffer has no bytes to push, try body_buffer
  NANOARROW_RETURN_NOT_OK(ArrowIpcArrayStreamWriterPush(private, &private->body_buffer,
                                                        &had_bytes_to_push, error));
  if (had_bytes_to_push) {
    return NANOARROW_OK;
  }

  // get the next Message
  if (private->schema.release == NULL) {
    // The schema message has not been buffered yet; do that now.
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayStreamGetSchema(&private->in, &private->schema, error));
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncoderEncodeSchema(&private->encoder, &private->schema, error));
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(&private->array_view, &private->schema, error));
  } else {
    // Get the next array from the stream
    NANOARROW_RETURN_NOT_OK(  // XXX does private->array maybe need to be released?
        ArrowArrayStreamGetNext(&private->in, &private->array, error));
    if (private->array.release == NULL) {
      // The stream is complete, signal the end to the caller
      writer->finished = 1;
      return NANOARROW_OK;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArray(&private->array_view, &private->array, error));

    NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeSimpleRecordBatch(
        &private->encoder, &private->array_view, &private->body_buffer, error));
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcEncoderFinalizeBuffer(&private->encoder, /*encapsulate=*/1,
                                    &private->buffer),
      error);
  NANOARROW_DCHECK(private->buffer.size_bytes % 8 == 0);
  NANOARROW_DCHECK(private->body_buffer.size_bytes % 8 == 0);
  // Since we just finalized it, buffer won't be empty
  return ArrowIpcArrayStreamWriterPush(private, &private->buffer, &had_bytes_to_push,
                                       error);
}
