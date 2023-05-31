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

#ifndef NANOARROW_DEVICE_METAL_H_INCLUDED
#define NANOARROW_DEVICE_METAL_H_INCLUDED

#include "nanoarrow_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup nanoarrow_device_metal Apple Metal Device extension
///
/// An Apple Metal implementation of the Arrow C Device interface, primarily targeted to
/// the M1 series of CPU/GPUs that feature shared CPU/GPU memory. Even though the memory
/// regions are shared, it is currently not possible to wrap an arbitrary CPU memory
/// region as an `MTL::Buffer*` unless that memory region is page-aligned. Because of
/// this, a copy is still required in most cases to make memory GPU accessible. After GPU
/// calculations are complete; however, moving the buffers back to the CPU is zero-copy.
///
/// Because of the buffer wrapping contraint, In this implementation, "GPU buffers" (i.e.,
/// the pointers in the ArrowArray buffers member) are represented as a pointer to an
/// `MTL::Buffer` instead of a pointer to its contents. This is necessary because the
/// methods that send a buffer to the GPU only accept an `MTL::Buffer*` (with offset +
/// length).
///
/// Sync events are represented as an `MTL::Event*`. The degree to which the pointers
/// to `MTL::Event*` and `MTL::Buffer*` are stable across metal-cpp versions/builds is
/// currently unknown.
///
/// @{

/// \brief A pointer to a default metal device singleton
struct ArrowDevice* ArrowDeviceMetalDefaultDevice(void);

/// \brief Initialize a preallocated device struct with the default metal device
ArrowErrorCode ArrowDeviceMetalInitDefaultDevice(struct ArrowDevice* device,
                                                 struct ArrowError* error);

/// \brief Initialize a buffer with the Metal allocator
///
/// Metal uses shared memory with the CPU; however, only page-aligned buffers
/// or buffers created explicitly using the Metal API can be sent to the GPU.
/// This buffer's allocator uses the Metal API so that it is cheaper to send
/// buffers to the GPU later. You can use, append to, or move this buffer just
/// like a normal ArrowBuffer.
ArrowErrorCode ArrowDeviceMetalInitCpuBuffer(struct ArrowDevice* device,
                                             struct ArrowBuffer* buffer,
                                             struct ArrowBufferView initial_content);

/// \brief Convert an ArrowArray to buffers that use the Metal allocator
///
/// Replaces buffers from a given ArrowArray with ones that use the Metal
/// allocator, copying existing content where necessary. The array is still
/// valid to use just like a normal ArrowArray that was initialized with
/// ArrowArrayInitFromType() (i.e., it can be appended to and finished with
/// validation).
ArrowErrorCode ArrowDeviceMetalInitCpuArrayBuffers(struct ArrowDevice* device,
                                                   struct ArrowArray* array);

/// @}

#ifdef __cplusplus
}
#endif

#endif
