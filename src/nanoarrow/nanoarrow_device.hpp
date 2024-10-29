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

#ifndef NANOARROW_DEVICE_HPP_INCLUDED
#define NANOARROW_DEVICE_HPP_INCLUDED

#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_device.h"

namespace nanoarrow {

namespace internal {

template <>
inline void init_pointer(struct ArrowDeviceArray* data) {
  data->array.release = nullptr;
  data->sync_event = nullptr;
}

template <>
inline void move_pointer(struct ArrowDeviceArray* src, struct ArrowDeviceArray* dst) {
  ArrowDeviceArrayMove(src, dst);
}

template <>
inline void release_pointer(struct ArrowDeviceArray* data) {
  if (data->array.release != nullptr) {
    ArrowArrayRelease(&data->array);
  }

  data->sync_event = nullptr;
}

template <>
inline void init_pointer(struct ArrowDeviceArrayStream* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowDeviceArrayStream* src,
                         struct ArrowDeviceArrayStream* dst) {
  memcpy(dst, src, sizeof(struct ArrowDeviceArrayStream));
  src->release = nullptr;
}

template <>
inline void release_pointer(struct ArrowDeviceArrayStream* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

template <>
inline void init_pointer(struct ArrowDeviceArrayView* data) {
  ArrowDeviceArrayViewInit(data);
}

template <>
inline void move_pointer(struct ArrowDeviceArrayView* src,
                         struct ArrowDeviceArrayView* dst) {
  ArrowArrayViewMove(&src->array_view, &dst->array_view);
  dst->device = src->device;
  src->device = nullptr;
}

template <>
inline void release_pointer(struct ArrowDeviceArrayView* data) {
  ArrowArrayViewReset(&data->array_view);
}

template <>
inline void init_pointer(struct ArrowDevice* data) {
  data->release = nullptr;
}

template <>
inline void move_pointer(struct ArrowDevice* src, struct ArrowDevice* dst) {
  memcpy(dst, src, sizeof(struct ArrowDevice));
  src->release = nullptr;
}

template <>
inline void release_pointer(struct ArrowDevice* data) {
  if (data->release != nullptr) {
    data->release(data);
  }
}

}  // namespace internal
}  // namespace nanoarrow

#include "nanoarrow.hpp"

namespace nanoarrow {

namespace device {

/// \defgroup nanoarrow_device_hpp-unique Unique object wrappers
///
/// Extends the unique object wrappers in nanoarrow.hpp to include C structs
/// defined in the nanoarrow_device.h header.
///
/// @{

/// \brief Class wrapping a unique struct ArrowDeviceArray
using UniqueDeviceArray = internal::Unique<struct ArrowDeviceArray>;

/// \brief Class wrapping a unique struct ArrowDeviceArrayStream
using UniqueDeviceArrayStream = internal::Unique<struct ArrowDeviceArrayStream>;

/// \brief Class wrapping a unique struct ArrowDevice
using UniqueDevice = internal::Unique<struct ArrowDevice>;

/// \brief Class wrapping a unique struct ArrowDeviceArrayView
using UniqueDeviceArrayView = internal::Unique<struct ArrowDeviceArrayView>;

/// @}

}  // namespace device

}  // namespace nanoarrow

#endif
