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

static inline uint32_t ArrowIpcReadUint32LE(struct ArrowIpcBufferView* data) {
  uint32_t value;
  memcpy(&value, data->data, sizeof(uint32_t));
  // bswap32() if big endian
  data->data += sizeof(uint32_t);
  data->size_bytes -= sizeof(uint32_t);
  return value;
}

static inline int32_t ArrowIpcReadInt32LE(struct ArrowIpcBufferView* data) {
  int32_t value;
  memcpy(&value, data->data, sizeof(int32_t));
  // bswap32() if big endian
  data->data += sizeof(int32_t);
  data->size_bytes -= sizeof(int32_t);
  return value;
}

#undef ns
#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

ArrowIpcErrorCode ArrowIpcDecodeMessage(struct ArrowIpcBufferView* data,
                                        int* message_type,
                                        struct ArrowArray* array_out,
                                        struct ArrowSchema* schema_out) {
  array_out->release = NULL;
  schema_out->release = NULL;
  *message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  struct ArrowIpcBufferView data_mut = *data;

  uint32_t continuation = ArrowIpcReadUint32LE(&data_mut);
  if (continuation != 0xFFFFFFFF) {
    return EINVAL;
  }

  int32_t message_size_bytes = ArrowIpcReadInt32LE(&data_mut);
  if (message_size_bytes > data_mut.size_bytes) {
    return ERANGE;
  }

  ns(Message_table_t) message = ns(Message_as_root(data_mut.data));
  if (!message) {
    return EINVAL;
  }

  int version = ns(Message_version(message));
  switch (version) {
    case ns(MetadataVersion_V5):
      break;
    default:
      return EINVAL;
  }

  *data = data_mut;
  return NANOARROW_OK;
}

ArrowIpcErrorCode ArrowIpcInitStreamReader(struct ArrowArrayStream* stream_out,
                                           struct ArrowIpcIO* io) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteSchema(struct ArrowArrayStream* stream_in,
                                      struct ArrowIpcIO* io) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteBatches(struct ArrowArrayStream* stream_in,
                                       struct ArrowIpcIO* io, int64_t num_batches) {
  return ENOTSUP;
}

ArrowIpcErrorCode ArrowIpcWriteEndOfStream(struct ArrowIpcIO* io) {
  return ENOTSUP;
}
