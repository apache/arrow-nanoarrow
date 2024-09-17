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

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/ipc/flatcc_generated.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

void ArrowIpcOutputStreamMove(struct ArrowIpcOutputStream* src,
                              struct ArrowIpcOutputStream* dst) {
  NANOARROW_DCHECK(src != NULL && dst != NULL);

  memcpy(dst, src, sizeof(struct ArrowIpcOutputStream));
  src->release = NULL;
}

ArrowErrorCode ArrowIpcOutputStreamWrite(struct ArrowIpcOutputStream* stream,
                                         struct ArrowBufferView data,
                                         struct ArrowError* error) {
  while (data.size_bytes != 0) {
    int64_t bytes_written = 0;
    NANOARROW_RETURN_NOT_OK(stream->write(stream, data.data.as_uint8, data.size_bytes,
                                          &bytes_written, error));
    data.size_bytes -= bytes_written;
    data.data.as_uint8 += bytes_written;
  }
  return NANOARROW_OK;
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
    return errno ? errno : EINVAL;
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

struct ArrowIpcWriterPrivate {
  struct ArrowIpcEncoder encoder;
  struct ArrowIpcOutputStream output_stream;
  struct ArrowBuffer buffer;
  struct ArrowBuffer body_buffer;

  int writing_file;
  int64_t bytes_written;
  struct ArrowIpcFooter footer;
};

ArrowErrorCode ArrowIpcWriterInit(struct ArrowIpcWriter* writer,
                                  struct ArrowIpcOutputStream* output_stream) {
  NANOARROW_DCHECK(writer != NULL && output_stream != NULL);

  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)ArrowMalloc(sizeof(struct ArrowIpcWriterPrivate));

  if (private == NULL) {
    return ENOMEM;
  }
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderInit(&private->encoder));
  ArrowIpcOutputStreamMove(output_stream, &private->output_stream);

  ArrowBufferInit(&private->buffer);
  ArrowBufferInit(&private->body_buffer);

  private->writing_file = 0;
  private->bytes_written = 0;
  ArrowIpcFooterInit(&private->footer);

  writer->private_data = private;
  return NANOARROW_OK;
}

void ArrowIpcWriterReset(struct ArrowIpcWriter* writer) {
  NANOARROW_DCHECK(writer != NULL);

  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)writer->private_data;

  if (private != NULL) {
    ArrowIpcEncoderReset(&private->encoder);
    private->output_stream.release(&private->output_stream);
    ArrowBufferReset(&private->buffer);
    ArrowBufferReset(&private->body_buffer);

    ArrowIpcFooterReset(&private->footer);

    ArrowFree(private);
  }
  memset(writer, 0, sizeof(struct ArrowIpcWriter));
}

static struct ArrowBufferView ArrowBufferToBufferView(const struct ArrowBuffer* buffer) {
  struct ArrowBufferView buffer_view = {
      .data.as_uint8 = buffer->data,
      .size_bytes = buffer->size_bytes,
  };
  return buffer_view;
}

// Eventually, it may be necessary to construct an ArrowIpcWriter which doesn't rely on
// blocking writes (ArrowIpcOutputStreamWrite). For example an ArrowIpcOutputStream
// might wrap a socket which is not always able to transmit all bytes of a Message. In
// that case users of ArrowIpcWriter might prefer to do other work until a socket is
// ready rather than blocking, or timeout, or otherwise respond to partial transmission.
//
// This could be handled by:
// - keeping partially sent buffers internal and signalling incomplete transmission by
//   raising EAGAIN, returning "bytes actually written", ...
//   - when the caller is ready to try again, call ArrowIpcWriterWriteSome()
// - exposing internal buffers which have not been completely sent, deferring
//   follow-up transmission to the caller

ArrowErrorCode ArrowIpcWriterWriteSchema(struct ArrowIpcWriter* writer,
                                         const struct ArrowSchema* in,
                                         struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL && in != NULL);
  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)writer->private_data;

  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->buffer, 0, 0));

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeSchema(&private->encoder, in, error));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcEncoderFinalizeBuffer(&private->encoder, /*encapsulate=*/1,
                                    &private->buffer),
      error);

  if (private->writing_file) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaDeepCopy(in, &private->footer.schema),
                                       error);
  }
  private->bytes_written += private->buffer.size_bytes;

  return ArrowIpcOutputStreamWrite(&private->output_stream,
                                   ArrowBufferToBufferView(&private->buffer), error);
}

ArrowErrorCode ArrowIpcWriterWriteArrayView(struct ArrowIpcWriter* writer,
                                            const struct ArrowArrayView* in,
                                            struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL);
  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)writer->private_data;

  if (in == NULL) {
    int32_t eos[] = {-1, 0};
    private->bytes_written += sizeof(eos);
    struct ArrowBufferView eos_view = {.data.as_int32 = eos, .size_bytes = sizeof(eos)};
    return ArrowIpcOutputStreamWrite(&private->output_stream, eos_view, error);
  }

  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->buffer, 0, 0));
  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->body_buffer, 0, 0));

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeSimpleRecordBatch(
      &private->encoder, in, &private->body_buffer, error));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcEncoderFinalizeBuffer(&private->encoder, /*encapsulate=*/1,
                                    &private->buffer),
      error);

  if (private->writing_file) {
    _NANOARROW_CHECK_RANGE(private->buffer.size_bytes, 0, INT32_MAX);
    struct ArrowIpcFileBlock block = {
        .offset = private->bytes_written,
        .metadata_length = (int32_t) private->buffer.size_bytes,
        .body_length = private->body_buffer.size_bytes,
    };
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(&private->footer.record_batch_blocks, &block, sizeof(block)),
        error);
  }
  private->bytes_written += private->buffer.size_bytes;
  private->bytes_written += private->body_buffer.size_bytes;

  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamWrite(
      &private->output_stream, ArrowBufferToBufferView(&private->buffer), error));
  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamWrite(
      &private->output_stream, ArrowBufferToBufferView(&private->body_buffer), error));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcWriterWriteArrayStreamImpl(
    struct ArrowIpcWriter* writer, struct ArrowArrayStream* in,
    struct ArrowSchema* schema, struct ArrowArray* array,
    struct ArrowArrayView* array_view, struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetSchema(in, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowIpcWriterWriteSchema(writer, schema, error));

  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(array_view, schema, error));
  while (1) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(in, array, error));
    if (array->release == NULL) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(array_view, array, error));
    NANOARROW_RETURN_NOT_OK(ArrowIpcWriterWriteArrayView(writer, array_view, error));
    ArrowArrayRelease(array);
  }

  // The stream is complete, signal the end to the caller
  return ArrowIpcWriterWriteArrayView(writer, NULL, error);
}

ArrowErrorCode ArrowIpcWriterWriteArrayStream(struct ArrowIpcWriter* writer,
                                              struct ArrowArrayStream* in,
                                              struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL && in != NULL);

  struct ArrowSchema schema = {.release = NULL};
  struct ArrowArray array = {.release = NULL};
  struct ArrowArrayView array_view;
  ArrowArrayViewInitFromType(&array_view, NANOARROW_TYPE_UNINITIALIZED);

  ArrowErrorCode result =
      ArrowIpcWriterWriteArrayStreamImpl(writer, in, &schema, &array, &array_view, error);

  if (schema.release != NULL) {
    ArrowSchemaRelease(&schema);
  }

  if (array.release != NULL) {
    ArrowArrayRelease(&array);
  }

  ArrowArrayViewReset(&array_view);

  return result;
}

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"

ArrowErrorCode ArrowIpcWriterStartFile(struct ArrowIpcWriter* writer,
                                       struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL);

  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)writer->private_data;

  NANOARROW_DCHECK(!private->writing_file && private->bytes_written == 0);

  struct ArrowBufferView magic = {
      .data.data = NANOARROW_IPC_FILE_PADDED_MAGIC,
      .size_bytes = sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC),
  };
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcOutputStreamWrite(&private->output_stream, magic, error));

  private->writing_file = 1;
  private->bytes_written = magic.size_bytes;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcWriterFinalizeFile(struct ArrowIpcWriter* writer,
                                          struct ArrowError* error) {
  NANOARROW_DCHECK(writer != NULL && writer->private_data != NULL);

  struct ArrowIpcWriterPrivate* private =
      (struct ArrowIpcWriterPrivate*)writer->private_data;

  NANOARROW_DCHECK(private->writing_file);

  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->buffer, 0, 0));
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncoderEncodeFooter(&private->encoder, &private->footer, error));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcEncoderFinalizeBuffer(&private->encoder, /*encapsulate=*/0,
                                    &private->buffer),
      error);

  _NANOARROW_CHECK_RANGE(private->buffer.size_bytes, 0, INT32_MAX);
  int32_t size = (int32_t) private->buffer.size_bytes;
  // we don't pad the magic at the end of the file
  struct ArrowStringView unpadded_magic = ArrowCharView(NANOARROW_IPC_FILE_PADDED_MAGIC);
  NANOARROW_DCHECK(unpadded_magic.size_bytes == 6);

  // just append to private->buffer instead of queueing two more tiny writes
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferReserve(&private->buffer, sizeof(size) + unpadded_magic.size_bytes),
      error);

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    size = (int32_t)bswap32((uint32_t)size);
  }
  NANOARROW_ASSERT_OK(ArrowBufferAppendInt32(&private->buffer, size));
  NANOARROW_ASSERT_OK(ArrowBufferAppendStringView(&private->buffer, unpadded_magic));

  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamWrite(
      &private->output_stream, ArrowBufferToBufferView(&private->buffer), error));
  private->bytes_written += private->buffer.size_bytes;
  return NANOARROW_OK;
}
