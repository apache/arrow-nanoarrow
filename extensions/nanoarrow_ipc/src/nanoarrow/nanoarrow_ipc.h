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

#ifndef NANOARROW_IPC_H_INCLUDED
#define NANOARROW_IPC_H_INCLUDED

#include "nanoarrow.h"

#ifdef NANOARROW_NAMESPACE

#define ArrowIpcCheckRuntime NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcCheckRuntime)
#define ArrowIpcReaderInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderInit)
#define ArrowIpcReaderReset NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderReset)
#define ArrowIpcReaderPeek NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderPeek)
#define ArrowIpcReaderVerify NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderVerify)
#define ArrowIpcReaderDecode NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderDecode)

#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ArrowIpcMessageType {
  NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED,
  NANOARROW_IPC_MESSAGE_TYPE_SCHEMA,
  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH,
  NANOARROW_IPC_MESSAGE_TYPE_TENSOR,
  NANOARROW_IPC_MESSAGE_TYPE_SPARSE_TENSOR
};

enum ArrowIpcEndianness {
  NANOARROW_IPC_ENDIANNESS_UNINITIALIZED,
  NANOARROW_IPC_ENDIANNESS_LITTLE,
  NANOARROW_IPC_ENDIANNESS_BIG
};

#define NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT 1
#define NANOARROW_IPC_FEATURE_COMPRESSED_BODY 2

ArrowErrorCode ArrowIpcCheckRuntime(struct ArrowError* error);

struct ArrowIpcReader {
  int32_t metadata_version;
  int32_t message_type;
  int32_t endianness;
  int32_t features;
  struct ArrowSchema schema;
};

void ArrowIpcReaderInit(struct ArrowIpcReader* reader);

void ArrowIpcReaderReset(struct ArrowIpcReader* reader);

ArrowErrorCode ArrowIpcReaderPeek(struct ArrowIpcReader* reader,
                                  struct ArrowBufferView* data, struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderVerify(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView* data,
                                    struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderDecode(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView* data,
                                    struct ArrowError* error);

#endif

#ifdef __cplusplus
}
#endif
