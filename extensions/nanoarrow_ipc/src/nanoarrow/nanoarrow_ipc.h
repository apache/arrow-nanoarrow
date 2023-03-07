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
#define ArrowIpcReaderGetSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderGetSchema)
#define ArrowIpcReaderGetArray \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderGetArray)
#define ArrowIpcReaderSetSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcReaderSetSchema)

#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ArrowIpcMetadataVersion {
  NANOARROW_IPC_METADATA_VERSION_V1,
  NANOARROW_IPC_METADATA_VERSION_V2,
  NANOARROW_IPC_METADATA_VERSION_V3,
  NANOARROW_IPC_METADATA_VERSION_V4,
  NANOARROW_IPC_METADATA_VERSION_V5
};

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

enum ArrowIpcCompressionType {
  NANOARROW_IPC_COMPRESSION_TYPE_NONE,
  NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
  NANOARROW_IPC_COMPRESSION_TYPE_ZSTD
};

#define NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT 1
#define NANOARROW_IPC_FEATURE_COMPRESSED_BODY 2

ArrowErrorCode ArrowIpcCheckRuntime(struct ArrowError* error);

/// \brief Decoder for Arrow IPC messages
///
/// This structure is intended to be allocated by the caller,
/// initialized using ArrowIpcReaderInit(), and released with
/// ArrowIpcReaderReset(). These fields should not be modified
/// by the caller but can be read following a call to
/// ArrowIpcReaderPeek(), ArrowIpcReaderVerify(), or
/// ArrowIpcReaderDecode().
struct ArrowIpcReader {
  /// \brief The last verified or decoded message type
  enum ArrowIpcMessageType message_type;

  /// \brief The metadata version used by this and forthcoming messages
  enum ArrowIpcMetadataVersion metadata_version;

  /// \brief Endianness of forthcoming RecordBatch messages
  enum ArrowIpcEndianness endianness;

  /// \brief Features used by this and forthcoming messages as indicated by the current
  /// Schema message
  int32_t feature_flags;

  /// \brief Compression used by the current RecordBatch message
  enum ArrowIpcCompressionType codec;

  /// \brief The number of bytes in the current header message
  ///
  /// This value includes the 8 bytes before the start of the header message
  /// content and any padding bytes required to make the header message size
  /// be a multiple of 8 bytes.
  int32_t header_size_bytes;

  /// \brief The number of bytes in the forthcoming body message.
  int64_t body_size_bytes;

  /// \brief Private resources managed by this library
  void* private_data;
};

ArrowErrorCode ArrowIpcReaderInit(struct ArrowIpcReader* reader);

void ArrowIpcReaderReset(struct ArrowIpcReader* reader);

ArrowErrorCode ArrowIpcReaderPeek(struct ArrowIpcReader* reader,
                                  struct ArrowBufferView data, struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderVerify(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderDecode(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderGetSchema(struct ArrowIpcReader* reader,
                                       struct ArrowSchema* out, struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderGetArray(struct ArrowIpcReader* reader,
                                      struct ArrowBufferView body, int64_t i,
                                      struct ArrowArray* out, struct ArrowError* error);

ArrowErrorCode ArrowIpcReaderSetSchema(struct ArrowIpcReader* reader,
                                       struct ArrowSchema* schema,
                                       struct ArrowError* error);

struct ArrowIpcField {
  struct ArrowArrayView* array_view;
  int64_t buffer_offset;
};

struct ArrowIpcReaderPrivate {
  struct ArrowSchema schema;
  struct ArrowArrayView array_view;
  int64_t n_fields;
  struct ArrowIpcField* fields;
  int64_t n_buffers;
  const void* last_message;
};

#ifdef __cplusplus
}
#endif

#endif
