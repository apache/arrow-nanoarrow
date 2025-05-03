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

#ifndef NANOARROW_IPC_HPP_INCLUDED
#define NANOARROW_IPC_HPP_INCLUDED

#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_ipc.h"

namespace nanoarrow {

namespace internal {

template <>
inline void init_pointer(struct ArrowIpcSharedBuffer* data) {
  init_pointer(&data->private_src);
}

template <>
inline void move_pointer(struct ArrowIpcSharedBuffer* src,
                         struct ArrowIpcSharedBuffer* dst) {
  move_pointer(&src->private_src, &dst->private_src);
}

template <>
inline void release_pointer(struct ArrowIpcSharedBuffer* data) {
  ArrowIpcSharedBufferReset(data);
}

template <>
inline void init_pointer(struct ArrowIpcDecoder* data) {
  data->private_data = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcDecoder* src, struct ArrowIpcDecoder* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcDecoder));
  src->private_data = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcDecoder* data) {
  ArrowIpcDecoderReset(data);
}

template <>
inline void init_pointer(struct ArrowIpcFooter* data) {
  ArrowIpcFooterInit(data);
}

template <>
inline void move_pointer(struct ArrowIpcFooter* src, struct ArrowIpcFooter* dst) {
  ArrowSchemaMove(&src->schema, &dst->schema);
  ArrowBufferMove(&src->record_batch_blocks, &dst->record_batch_blocks);
}

template <>
inline void release_pointer(struct ArrowIpcFooter* data) {
  ArrowIpcFooterReset(data);
}

template <>
inline void init_pointer(struct ArrowIpcEncoder* data) {
  data->private_data = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcEncoder* src, struct ArrowIpcEncoder* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcEncoder));
  src->private_data = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcEncoder* data) {
  ArrowIpcEncoderReset(data);
}

template <>
inline void init_pointer(struct ArrowIpcDecompressor* data) {
  data->private_data = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcDecompressor* src,
                         struct ArrowIpcDecompressor* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcDecompressor));
  src->release = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcDecompressor* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowIpcInputStream* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcInputStream* src,
                         struct ArrowIpcInputStream* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcInputStream));
  src->release = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcInputStream* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowIpcOutputStream* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcOutputStream* src,
                         struct ArrowIpcOutputStream* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcOutputStream));
  src->release = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcOutputStream* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowIpcWriter* data) {
  data->private_data = nullptr;
}

template <>
inline void move_pointer(struct ArrowIpcWriter* src, struct ArrowIpcWriter* dst) {
  memcpy(dst, src, sizeof(struct ArrowIpcWriter));
  src->private_data = nullptr;
}

template <>
inline void release_pointer(struct ArrowIpcWriter* data) {
  ArrowIpcWriterReset(data);
}

}  // namespace internal
}  // namespace nanoarrow

namespace nanoarrow {

namespace ipc {

/// \defgroup nanoarrow_ipc_hpp-unique Unique object wrappers
///
/// Extends the unique object wrappers in nanoarrow.hpp to include C structs
/// defined in the nanoarrow_ipc.h header.
///
/// @{

/// \brief Class wrapping a unique struct ArrowIpcSharedBuffer
using UniqueSharedBuffer = internal::Unique<struct ArrowIpcSharedBuffer>;

/// \brief Class wrapping a unique struct ArrowIpcDecoder
using UniqueDecoder = internal::Unique<struct ArrowIpcDecoder>;

/// \brief Class wrapping a unique struct ArrowIpcFooter
using UniqueFooter = internal::Unique<struct ArrowIpcFooter>;

/// \brief Class wrapping a unique struct ArrowIpcEncoder
using UniqueEncoder = internal::Unique<struct ArrowIpcEncoder>;

/// \brief Class wrapping a unique struct ArrowIpcDecompressor
using UniqueDecompressor = internal::Unique<struct ArrowIpcDecompressor>;

/// \brief Class wrapping a unique struct ArrowIpcInputStream
using UniqueInputStream = internal::Unique<struct ArrowIpcInputStream>;

/// \brief Class wrapping a unique struct ArrowIpcOutputStream
using UniqueOutputStream = internal::Unique<struct ArrowIpcOutputStream>;

/// \brief Class wrapping a unique struct ArrowIpcWriter
using UniqueWriter = internal::Unique<struct ArrowIpcWriter>;

/// @}

}  // namespace ipc

}  // namespace nanoarrow

#endif
