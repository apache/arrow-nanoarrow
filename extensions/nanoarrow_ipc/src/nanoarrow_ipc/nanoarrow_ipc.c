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
#include <string.h>

#include "nanoarrow/nanoarrow.h"

#include "File_builder.h"
#include "File_reader.h"
#include "Message_builder.h"
#include "Message_reader.h"
#include "Schema_builder.h"
#include "Schema_reader.h"

#include "nanoarrow_ipc.h"

struct ArrowIpcBufferViewReaderPrivate {
  const uint8_t* data0;
  const uint8_t* data;
  int64_t size_bytes;
};

static ArrowIpcErrorCode ArrowIpcIOReadVoid(struct ArrowIpcIO* io, uint8_t* dst,
                                            int64_t dst_size, int64_t* size_read_out) {
  return ENOTSUP;
}

static ArrowIpcErrorCode ArrowIpcIOWriteVoid(struct ArrowIpcIO* io, const uint8_t* src,
                                             int64_t src_size) {
  return ENOTSUP;
}

static ArrowIpcErrorCode ArrowIpcIOSizeVoid(struct ArrowIpcIO* io, int64_t* size_out) {
  return ENOTSUP;
}

static ArrowIpcErrorCode ArrowIpcIOSeekVoid(struct ArrowIpcIO* io, int64_t position) {
  return ENOTSUP;
}

static const char* ArrowIpcIOGetLastErrorVoid(struct ArrowIpcIO* io) { return NULL; }

static void ArrowIpcIOReleaseVoid(struct ArrowIpcIO* io) { io->release = NULL; }

static void ArrowIpcIOInitVoid(struct ArrowIpcIO* io) {
  io->read = &ArrowIpcIOReadVoid;
  io->write = &ArrowIpcIOWriteVoid;
  io->size = &ArrowIpcIOSizeVoid;
  io->seek = &ArrowIpcIOSeekVoid;
  io->release = &ArrowIpcIOReleaseVoid;
  io->private_data = NULL;
}

static ArrowIpcErrorCode ArrowIpcIOReadBufferView(struct ArrowIpcIO* io, uint8_t* dst,
                                                  int64_t dst_size,
                                                  int64_t* size_read_out) {
  if (dst_size == 0) {
    return NANOARROW_OK;
  }

  struct ArrowIpcBufferViewReaderPrivate* private_data =
      (struct ArrowIpcBufferViewReaderPrivate*)io->private_data;

  if (private_data->size_bytes > dst_size) {
    *size_read_out = dst_size;
  } else {
    *size_read_out = private_data->size_bytes;
  }

  memcpy(dst, private_data->data, *size_read_out);
  private_data->data += *size_read_out;
  private_data->size_bytes -= *size_read_out;
  return NANOARROW_OK;
}

static ArrowIpcErrorCode ArrowIpcIOSizeBufferView(struct ArrowIpcIO* io, int64_t* size_out) {
  struct ArrowIpcBufferViewReaderPrivate* private_data =
      (struct ArrowIpcBufferViewReaderPrivate*)io->private_data;

  *size_out = (private_data->data - private_data->data0) + private_data->size_bytes;
  return NANOARROW_OK;
}

static ArrowIpcErrorCode ArrowIpcIOSeekBufferView(struct ArrowIpcIO* io, int64_t position) {
  struct ArrowIpcBufferViewReaderPrivate* private_data =
      (struct ArrowIpcBufferViewReaderPrivate*)io->private_data;

  int64_t size = (private_data->data - private_data->data0) + private_data->size_bytes;
  if (position < 0 || position > size) {
    return ERANGE;
  }

  private_data->size_bytes += size - position;
  private_data->data = private_data->data0 + position;

  return NANOARROW_OK;
}

static void ArrowIpcIOReleaseBufferView(struct ArrowIpcIO* io) {
  ArrowFree(io->private_data);
  io->release = NULL;
}

ArrowIpcErrorCode ArrowIpcInitBufferViewReader(struct ArrowIpcIO* io, const void* data,
                                               int64_t size_bytes) {
  ArrowIpcIOInitVoid(io);

  struct ArrowIpcBufferViewReaderPrivate* private_data =
      (struct ArrowIpcBufferViewReaderPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcBufferViewReaderPrivate));

  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->data0 = data;
  private_data->data = data;
  private_data->size_bytes = size_bytes;

  io->read = &ArrowIpcIOReadBufferView;
  io->size = &ArrowIpcIOSizeBufferView;
  io->seek = &ArrowIpcIOSeekBufferView;
  io->release = &ArrowIpcIOReleaseBufferView;
  io->private_data = private_data;

  return NANOARROW_OK;
}

ArrowIpcErrorCode ArrowIpcInitStreamReader(struct ArrowArrayStream* stream_out,
                                           struct ArrowIpcIO* io,
                                           struct ArrowIpcError* error) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteSchema(struct ArrowArrayStream* stream_in,
                                      struct ArrowIpcIO* io,
                                      struct ArrowIpcError* error) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteBatches(struct ArrowArrayStream* stream_in,
                                       struct ArrowIpcIO* io, int64_t num_batches,
                                       struct ArrowIpcError* error) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteEndOfStream(struct ArrowIpcIO* io,
                                           struct ArrowIpcError* error) {
  return ENOTSUP;
}
