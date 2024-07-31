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
