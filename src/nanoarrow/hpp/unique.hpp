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

#ifndef NANOARROW_HPP_UNIQUE_HPP_INCLUDED
#define NANOARROW_HPP_UNIQUE_HPP_INCLUDED

#include <string.h>

#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

namespace internal {

/// \defgroup nanoarrow_hpp-unique_base Base classes for Unique wrappers
///
/// @{

template <typename T>
static inline void init_pointer(T* data);

template <typename T>
static inline void move_pointer(T* src, T* dst);

template <typename T>
static inline void release_pointer(T* data);

template <>
inline void init_pointer(struct ArrowSchema* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowSchema* src, struct ArrowSchema* dst) {
  ArrowSchemaMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowSchema* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowArray* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowArray* src, struct ArrowArray* dst) {
  ArrowArrayMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowArray* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowArrayStream* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowArrayStream* src, struct ArrowArrayStream* dst) {
  ArrowArrayStreamMove(src, dst);
}

template <>
inline void release_pointer(ArrowArrayStream* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowBuffer* data) {
  ArrowBufferInit(data);
}

template <>
inline void move_pointer(struct ArrowBuffer* src, struct ArrowBuffer* dst) {
  ArrowBufferMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowBuffer* data) {
  ArrowBufferReset(data);
}

template <>
inline void init_pointer(struct ArrowBitmap* data) {
  ArrowBitmapInit(data);
}

template <>
inline void move_pointer(struct ArrowBitmap* src, struct ArrowBitmap* dst) {
  ArrowBitmapMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowBitmap* data) {
  ArrowBitmapReset(data);
}

template <>
inline void init_pointer(struct ArrowArrayView* data) {
  ArrowArrayViewInitFromType(data, NANOARROW_TYPE_UNINITIALIZED);
}

template <>
inline void move_pointer(struct ArrowArrayView* src, struct ArrowArrayView* dst) {
  ArrowArrayViewMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowArrayView* data) {
  ArrowArrayViewReset(data);
}

/// \brief A unique_ptr-like base class for stack-allocatable objects
/// \tparam T The object type
template <typename T>
class Unique {
 public:
  /// \brief Construct an invalid instance of T holding no resources
  Unique() {
    memset(&data_, 0, sizeof(data_));
    init_pointer(&data_);
  }

  /// \brief Move and take ownership of data
  Unique(T* data) {
    memset(&data_, 0, sizeof(data_));
    move_pointer(data, &data_);
  }

  /// \brief Move and take ownership of data wrapped by rhs
  Unique(Unique&& rhs) : Unique(rhs.get()) {}
  Unique& operator=(Unique&& rhs) {
    reset(rhs.get());
    return *this;
  }

  // These objects are not copyable
  Unique(const Unique& rhs) = delete;

  /// \brief Get a pointer to the data owned by this object
  T* get() noexcept { return &data_; }
  const T* get() const noexcept { return &data_; }

  /// \brief Use the pointer operator to access fields of this object
  T* operator->() noexcept { return &data_; }
  const T* operator->() const noexcept { return &data_; }

  /// \brief Call data's release callback if valid
  void reset() { release_pointer(&data_); }

  /// \brief Call data's release callback if valid and move ownership of the data
  /// pointed to by data
  void reset(T* data) {
    reset();
    move_pointer(data, &data_);
  }

  /// \brief Move ownership of this object to the data pointed to by out
  void move(T* out) { move_pointer(&data_, out); }

  ~Unique() { reset(); }

 protected:
  T data_;
};

/// @}

}  // namespace internal

/// \defgroup nanoarrow_hpp-unique Unique object wrappers
///
/// The Arrow C Data interface, the Arrow C Stream interface, and the
/// nanoarrow C library use stack-allocatable objects, some of which
/// require initialization or cleanup.
///
/// @{

/// \brief Class wrapping a unique struct ArrowSchema
using UniqueSchema = internal::Unique<struct ArrowSchema>;

/// \brief Class wrapping a unique struct ArrowArray
using UniqueArray = internal::Unique<struct ArrowArray>;

/// \brief Class wrapping a unique struct ArrowArrayStream
using UniqueArrayStream = internal::Unique<struct ArrowArrayStream>;

/// \brief Class wrapping a unique struct ArrowBuffer
using UniqueBuffer = internal::Unique<struct ArrowBuffer>;

/// \brief Class wrapping a unique struct ArrowBitmap
using UniqueBitmap = internal::Unique<struct ArrowBitmap>;

/// \brief Class wrapping a unique struct ArrowArrayView
using UniqueArrayView = internal::Unique<struct ArrowArrayView>;

/// @}

NANOARROW_CXX_NAMESPACE_END

#endif
