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

struct ArrowIpcInputStreamLiteralPrivate {
  struct ArrowBuffer input;
  int64_t cursor_bytes;
};

static ArrowErrorCode ArrowIpcInputStreamLiteralRead(struct ArrowIpcInputStream* stream,
                                                     void* buf, int64_t buf_size_bytes,
                                                     int64_t* size_read_out,
                                                     struct ArrowError* error) {
  if (buf_size_bytes == 0) {
    return NANOARROW_OK;
  }

  struct ArrowIpcInputStreamLiteralPrivate* private_data =
      (struct ArrowIpcInputStreamLiteralPrivate*)stream->private_data;
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

static void ArrowIpcInputStreamLiteralRelease(struct ArrowIpcInputStream* stream) {
  struct ArrowIpcInputStreamLiteralPrivate* private_data =
      (struct ArrowIpcInputStreamLiteralPrivate*)stream->private_data;
  ArrowBufferReset(&private_data->input);
  ArrowFree(private_data);
  stream->release = NULL;
}

ArrowErrorCode ArrowIpcInputStreamInitLiteral(struct ArrowIpcInputStream* stream,
                                              struct ArrowBuffer* input) {
  struct ArrowIpcInputStreamLiteralPrivate* private_data =
      (struct ArrowIpcInputStreamLiteralPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcInputStreamLiteralPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  ArrowBufferMove(input, &private_data->input);
  private_data->cursor_bytes = 0;
  stream->read = &ArrowIpcInputStreamLiteralRead;
  stream->release = &ArrowIpcInputStreamLiteralRelease;
  stream->private_data = private_data;

  return NANOARROW_OK;
}
