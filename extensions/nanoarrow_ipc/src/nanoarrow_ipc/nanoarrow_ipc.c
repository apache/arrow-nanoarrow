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

#define ArrowIpcErrorSet(err, ...) ArrowErrorSet((struct ArrowError*)err, __VA_ARGS__)

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
                                        int* message_type, struct ArrowArray* array_out,
                                        struct ArrowSchema* schema_out,
                                        struct ArrowIpcError* error) {
  array_out->release = NULL;
  schema_out->release = NULL;
  *message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  struct ArrowIpcBufferView data_mut = *data;

  if (data_mut.size_bytes < 8) {
    ArrowIpcErrorSet(error,
                     "Expected message of at least 8 bytes but only %ld bytes remain",
                     (long)data->size_bytes);
  }

  uint32_t continuation = ArrowIpcReadUint32LE(&data_mut);
  if (continuation != 0xFFFFFFFF) {
    ArrowIpcErrorSet(error, "Expected 0xFFFFFFFF at start of message but found %dx",
                     (unsigned int)continuation);
    return EINVAL;
  }

  int32_t message_size_bytes = ArrowIpcReadInt32LE(&data_mut);
  if (message_size_bytes > data_mut.size_bytes) {
    ArrowIpcErrorSet(
        error, "Expected message size >= 0 bytes but found message size of %ld bytes",
        (long)message_size_bytes);
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
      ArrowIpcErrorSet(error, "Expected metadata version V5(4L) but found %d", version);
      return EINVAL;
  }

  *message_type = ns(Message_header_type(message));
  switch (*message_type) {
    case ns(MessageHeader_Schema):
    case ns(MessageHeader_DictionaryBatch):
    case ns(MessageHeader_RecordBatch):
    case ns(MessageHeader_Tensor):
    case ns(MessageHeader_SparseTensor):
      ArrowIpcErrorSet(error, "Unsupported message type: '%s'",
                       ns(MessageHeader_type_name(*message_type)));
      return ENOTSUP;
    default:
      ArrowIpcErrorSet(error, "Unnown message type: %d", (int)(*message_type));
      return EINVAL;
  }

  *data = data_mut;
  return NANOARROW_OK;
}
