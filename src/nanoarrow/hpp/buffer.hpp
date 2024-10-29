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

#ifndef NANOARROW_HPP_BUFFER_HPP_INCLUDED
#define NANOARROW_HPP_BUFFER_HPP_INCLUDED

#include <stdint.h>
#include <utility>
#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

namespace internal {
template <typename T>
static inline void DeallocateWrappedBuffer(struct ArrowBufferAllocator* allocator,
                                           uint8_t* ptr, int64_t size) {
  NANOARROW_UNUSED(ptr);
  NANOARROW_UNUSED(size);
  auto obj = reinterpret_cast<T*>(allocator->private_data);
  delete obj;
}
}  // namespace internal

/// \defgroup nanoarrow_hpp-buffer Buffer helpers
///
/// Helpers to wrap buffer-like C++ objects as ArrowBuffer objects that can
/// be used to build ArrowArray objects.
///
/// @{

/// \brief Initialize a buffer wrapping an arbitrary C++ object
///
/// Initializes a buffer with a release callback that deletes the moved obj
/// when ArrowBufferReset is called. This version is useful for wrapping
/// an object whose .data() member is missing or unrelated to the buffer
/// value that is destined for a the buffer of an ArrowArray. T must be movable.
template <typename T>
static inline void BufferInitWrapped(struct ArrowBuffer* buffer, T obj,
                                     const uint8_t* data, int64_t size_bytes) {
  T* obj_moved = new T(std::move(obj));
  buffer->data = const_cast<uint8_t*>(data);
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = 0;
  buffer->allocator =
      ArrowBufferDeallocator(&internal::DeallocateWrappedBuffer<T>, obj_moved);
}

/// \brief Initialize a buffer wrapping a C++ sequence
///
/// Specifically, this uses obj.data() to set the buffer address and
/// obj.size() * sizeof(T::value_type) to set the buffer size. This works
/// for STL containers like std::vector, std::array, and std::string.
/// This function moves obj and ensures it is deleted when ArrowBufferReset
/// is called.
template <typename T>
void BufferInitSequence(struct ArrowBuffer* buffer, T obj) {
  // Move before calling .data() (matters sometimes).
  T* obj_moved = new T(std::move(obj));
  buffer->data =
      const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj_moved->data()));
  buffer->size_bytes = obj_moved->size() * sizeof(typename T::value_type);
  buffer->capacity_bytes = 0;
  buffer->allocator =
      ArrowBufferDeallocator(&internal::DeallocateWrappedBuffer<T>, obj_moved);
}

/// @}

NANOARROW_CXX_NAMESPACE_END

#endif
