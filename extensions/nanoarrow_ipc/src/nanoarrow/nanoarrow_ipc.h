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
#define ArrowIpcDecoderInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderInit)
#define ArrowIpcDecoderReset NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderReset)
#define ArrowIpcDecoderPeek NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderPeek)
#define ArrowIpcDecoderVerify NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderVerify)
#define ArrowIpcDecoderDecode NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderDecode)
#define ArrowIpcDecoderGetSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderGetSchema)
#define ArrowIpcDecoderGetArray \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderGetArray)
#define ArrowIpcDecoderSetSchema \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderSetSchema)

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
/// initialized using ArrowIpcDecoderInit(), and released with
/// ArrowIpcDecoderReset(). These fields should not be modified
/// by the caller but can be read following a call to
/// ArrowIpcDecoderPeek(), ArrowIpcDecoderVerify(), or
/// ArrowIpcDecoderDecode().
struct ArrowIpcDecoder {
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

ArrowErrorCode ArrowIpcDecoderInit(struct ArrowIpcDecoder* decoder);

void ArrowIpcDecoderReset(struct ArrowIpcDecoder* decoder);

ArrowErrorCode ArrowIpcDecoderPeek(struct ArrowIpcDecoder* decoder,
                                   struct ArrowBufferView data, struct ArrowError* error);

ArrowErrorCode ArrowIpcDecoderVerify(struct ArrowIpcDecoder* decoder,
                                     struct ArrowBufferView data,
                                     struct ArrowError* error);

ArrowErrorCode ArrowIpcDecoderDecode(struct ArrowIpcDecoder* decoder,
                                     struct ArrowBufferView data,
                                     struct ArrowError* error);

ArrowErrorCode ArrowIpcDecoderGetSchema(struct ArrowIpcDecoder* decoder,
                                        struct ArrowSchema* out,
                                        struct ArrowError* error);

ArrowErrorCode ArrowIpcDecoderGetArray(struct ArrowIpcDecoder* decoder,
                                       struct ArrowBufferView body, int64_t i,
                                       struct ArrowArray* out, struct ArrowError* error);

ArrowErrorCode ArrowIpcDecoderSetSchema(struct ArrowIpcDecoder* decoder,
                                        struct ArrowSchema* schema,
                                        struct ArrowError* error);

struct ArrowIpcField {
  struct ArrowArrayView* array_view;
  int64_t buffer_offset;
};

struct ArrowIpcDecoderPrivate {
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
