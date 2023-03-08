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
#define ArrowIpcDecoderDecodeHeader \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowIpcDecoderDecodeHeader)
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
/// ArrowIpcDecoderDecodeHeader().
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

/// \brief Initialize a decoder
ArrowErrorCode ArrowIpcDecoderInit(struct ArrowIpcDecoder* decoder);

/// \brief Release all resources attached to a decoder
void ArrowIpcDecoderReset(struct ArrowIpcDecoder* decoder);

/// \brief Peek at a message header
///
/// The first 8 bytes of an Arrow IPC message are 0xFFFFFF followed by the size
/// of the header as a little-endian 32-bit integer. ArrowIpcDecoderPeek() reads
/// these bytes and returns ESPIPE if there are not enough remaining bytes in data to read
/// the entire header message, EINVAL if the first 8 bytes are not valid, ENODATA if the
/// Arrow end-of-stream indicator has been reached, or NANOARROW_OK otherwise.
ArrowErrorCode ArrowIpcDecoderPeek(struct ArrowIpcDecoder* decoder,
                                   struct ArrowBufferView data, struct ArrowError* error);

/// \brief Verify a message header
///
/// Runs ArrowIpcDecoderPeek() to ensure data is sufficiently large but additionally
/// runs flatbuffer verification to ensure that decoding the data will not access
/// memory outside of the buffer specified by data. ArrowIpcDecoderVerify() will also
/// set decoder.header_size_bytes, decoder.body_size_bytes, decoder.metadata_version,
/// and decoder.message_type.
///
/// Returns as ArrowIpcDecoderPeek() and additionally will
/// return EINVAL if flatbuffer verification fails.
ArrowErrorCode ArrowIpcDecoderVerify(struct ArrowIpcDecoder* decoder,
                                     struct ArrowBufferView data,
                                     struct ArrowError* error);

/// \brief Decode a message header
///
/// Runs ArrowIpcDecoderPeek() to ensure data is sufficiently large and decodes
/// the content of the message header. If data contains a schema message,
/// decoder.endianness and decoder.feature_flags is set and ArrowIpcDecoderGetSchema() can
/// be used to obtain the decoded schema. If data contains a record batch message,
/// decoder.codec is set and a successful call can be followed by a call to
/// ArrowIpcDecoderGetArray().
///
/// In almost all cases this should be preceeded by a call to ArrowIpcDecoderVerify() to
/// ensure decoding does not access data outside of the specified buffer.
///
/// Returns EINVAL if the content of the message cannot be decoded or ENOTSUP if the
/// content of the message uses features not supported by this library.
ArrowErrorCode ArrowIpcDecoderDecodeHeader(struct ArrowIpcDecoder* decoder,
                                           struct ArrowBufferView data,
                                           struct ArrowError* error);

/// \brief Decode an ArrowArray
///
/// After a successful call to ArrowIpcDecoderDecodeHeader(), retrieve an ArrowSchema.
/// The caller is responsible for releasing the schema if NANOARROW_OK is returned.
///
/// Returns EINVAL if the decoder did not just decode a schema message or
/// NANOARROW_OK otherwise.
ArrowErrorCode ArrowIpcDecoderGetSchema(struct ArrowIpcDecoder* decoder,
                                        struct ArrowSchema* out,
                                        struct ArrowError* error);

/// \brief Set the ArrowSchema used to decode future record batch messages
///
/// Prepares the decoder for future record batch messages
/// of this type. The decoder takes ownership of schema if NANOARROW_OK is returned.
///
/// Returns EINVAL if schema validation fails or NANOARROW_OK otherwise.
ArrowErrorCode ArrowIpcDecoderSetSchema(struct ArrowIpcDecoder* decoder,
                                        struct ArrowSchema* schema,
                                        struct ArrowError* error);

/// \brief Decode an ArrowArray
///
/// After a successful call to ArrowIpcDecoderDecodeHeader(), assemble an ArrowArray given
/// a message body and a field index. Note that field index does not equate to column
/// index if any columns contain nested types. Use a value of -1 to decode the entire
/// array into a struct. The caller is responsible for releasing the array if
/// NANOARROW_OK is returned.
///
/// Returns EINVAL if the decoder did not just decode a record batch message, ENOTSUP
/// if the message uses features not supported by this library, or or NANOARROW_OK
/// otherwise.
ArrowErrorCode ArrowIpcDecoderGetArray(struct ArrowIpcDecoder* decoder,
                                       struct ArrowBufferView body, int64_t i,
                                       struct ArrowArray* out, struct ArrowError* error);

#ifdef __cplusplus
}
#endif

#endif
