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

#ifndef NANOARROW_DEVICE_H_INCLUDED
#define NANOARROW_DEVICE_H_INCLUDED

#include "nanoarrow.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup nanoarrow_device-arrow-cdata Arrow C Device interface
///
/// The Arrow Device and Stream interfaces are part of the
/// Arrow C Device Data and Arrow C Device stream interfaces
/// (https://arrow.apache.org/docs/dev/format/CDeviceDataInterface.html).
/// See the Arrow documentation for detailed documentation of these structures.
///
/// @{

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

// Device type for the allocated memory
typedef int32_t ArrowDeviceType;

// CPU device, same as using ArrowArray directly
#define ARROW_DEVICE_CPU 1
// CUDA GPU Device
#define ARROW_DEVICE_CUDA 2
// Pinned CUDA CPU memory by cudaMallocHost
#define ARROW_DEVICE_CUDA_HOST 3
// OpenCL Device
#define ARROW_DEVICE_OPENCL 4
// Vulkan buffer for next-gen graphics
#define ARROW_DEVICE_VULKAN 7
// Metal for Apple GPU
#define ARROW_DEVICE_METAL 8
// Verilog simulator buffer
#define ARROW_DEVICE_VPI 9
// ROCm GPUs for AMD GPUs
#define ARROW_DEVICE_ROCM 10
// Pinned ROCm CPU memory allocated by hipMallocHost
#define ARROW_DEVICE_ROCM_HOST 11
// Reserved for extension
//
// used to quickly test extension devices, semantics
// can differ based on implementation
#define ARROW_DEVICE_EXT_DEV 12
// CUDA managed/unified memory allocated by cudaMallocManaged
#define ARROW_DEVICE_CUDA_MANAGED 13
// Unified shared memory allocated on a oneAPI
// non-partitioned device.
//
// A call to the oneAPI runtime is required to determine the
// device type, the USM allocation type and the sycl context
// that it is bound to.
#define ARROW_DEVICE_ONEAPI 14
// GPU support for next-gen WebGPU standard
#define ARROW_DEVICE_WEBGPU 15
// Qualcomm Hexagon DSP
#define ARROW_DEVICE_HEXAGON 16

struct ArrowDeviceArray {
  struct ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  void* sync_event;

  // reserved bytes for future expansion
  int64_t reserved[3];
};

#endif  // ARROW_C_DEVICE_DATA_INTERFACE

#ifndef ARROW_C_DEVICE_STREAM_INTERFACE
#define ARROW_C_DEVICE_STREAM_INTERFACE

struct ArrowDeviceArrayStream {
  // device type that all arrays will be accessible from
  ArrowDeviceType device_type;
  // callbacks
  int (*get_schema)(struct ArrowDeviceArrayStream*, struct ArrowSchema*);
  int (*get_next)(struct ArrowDeviceArrayStream*, struct ArrowDeviceArray*);
  const char* (*get_last_error)(struct ArrowDeviceArrayStream*);

  // release callback
  void (*release)(struct ArrowDeviceArrayStream*);

  // opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DEVICE_STREAM_INTERFACE

/// \brief Move the contents of src into dst and set src->array.release to NULL
static inline void ArrowDeviceArrayMove(struct ArrowDeviceArray* src,
                                        struct ArrowDeviceArray* dst) {
  memcpy(dst, src, sizeof(struct ArrowDeviceArray));
  src->array.release = 0;
}

/// @}

#ifdef NANOARROW_NAMESPACE

#define ArrowDeviceCheckRuntime \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceCheckRuntime)
#define ArrowDeviceArrayInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayInit)
#define ArrowDeviceArrayViewInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewInit)
#define ArrowDeviceArrayViewReset \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewReset)
#define ArrowDeviceArrayViewSetArray \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewSetArray)
#define ArrowDeviceArrayViewCopy \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewCopy)
#define ArrowDeviceArrayViewCopyRequired \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewCopyRequired)
#define ArrowDeviceArrayTryMove \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayTryMove)
#define ArrowDeviceResolve NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceResolve)
#define ArrowDeviceCpu NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceCpu)
#define ArrowDeviceInitCpu NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceInitCpu)
#define ArrowDeviceBufferInit NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferInit)
#define ArrowDeviceBufferMove NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferMove)
#define ArrowDeviceBufferCopy NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferCopy)
#define ArrowDeviceBasicArrayStreamInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBasicArrayStreamInit)

#endif

/// \defgroup nanoarrow_device Nanoarrow Device extension
///
/// Except where noted, objects are not thread-safe and clients should
/// take care to serialize accesses to methods.
///
/// @{

/// \brief Checks the nanoarrow runtime to make sure the run/build versions match
ArrowErrorCode ArrowDeviceCheckRuntime(struct ArrowError* error);

/// \brief A description of a buffer
struct ArrowDeviceBufferView {
  /// \brief Device-defined handle for a buffer.
  ///
  /// For the CPU device, this is a normal memory address; for all other types that are
  /// currently supported, this is a device memory address on which CPU-like arithmetic
  /// can be performed. This may not be true for future devices (i.e., it may be a pointer
  /// to some buffer abstraction if the concept of a memory address does not exist or
  /// is impractical).
  const void* private_data;

  /// \brief An offset into the buffer handle defined by private_data
  int64_t offset_bytes;

  /// \brief The size of the buffer in bytes
  int64_t size_bytes;
};

/// \brief A Device wrapper with callbacks for basic memory management tasks
///
/// All device objects are currently implemented as singletons; however, this
/// may change as implementations progress.
struct ArrowDevice {
  /// \brief The device type integer identifier (see ArrowDeviceArray)
  ArrowDeviceType device_type;

  /// \brief The device identifier (see ArrowDeviceArray)
  int64_t device_id;

  /// \brief Initialize an owning buffer from existing content
  ///
  /// Creates a new buffer whose data member can be accessed by the GPU by
  /// copying existing content.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_init)(struct ArrowDevice* device_src,
                                struct ArrowDeviceBufferView src,
                                struct ArrowDevice* device_dst, struct ArrowBuffer* dst);

  /// \brief Move an owning buffer to a device
  ///
  /// Creates a new buffer whose data member can be accessed by the GPU by
  /// moving an existing buffer. If NANOARROW_OK is returned, src will have
  /// been released or moved by the implementation and dst must be released by
  /// the caller.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_move)(struct ArrowDevice* device_src, struct ArrowBuffer* src,
                                struct ArrowDevice* device_dst, struct ArrowBuffer* dst);

  /// \brief Copy a section of memory into a preallocated buffer
  ///
  /// As opposed to the other buffer operations, this is designed to support
  /// copying very small slices of memory.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_copy)(struct ArrowDevice* device_src,
                                struct ArrowDeviceBufferView src,
                                struct ArrowDevice* device_dst,
                                struct ArrowDeviceBufferView dst);

  /// \brief Check if a copy is required to move between devices
  ///
  /// Returns 1 (copy is required), 0 (copy not required; move is OK), or -1 (don't know)
  int (*copy_required)(struct ArrowDevice* device_src, struct ArrowArrayView* src,
                       struct ArrowDevice* device_dst);

  /// \brief Wait for an event
  ///
  /// Implementations should handle at least waiting on the CPU host.
  /// Implementations do not have to handle a NULL sync_event.
  ArrowErrorCode (*synchronize_event)(struct ArrowDevice* device,
                                      struct ArrowDevice* device_event, void* sync_event,
                                      struct ArrowError* error);

  /// \brief Release this device and any resources it holds
  void (*release)(struct ArrowDevice* device);

  /// \brief Opaque, implementation-specific data.
  void* private_data;
};

struct ArrowDeviceArrayView {
  struct ArrowDevice* device;
  struct ArrowArrayView array_view;
};

/// \brief Initialize an ArrowDeviceArray
///
/// Zeroes the memory of device_array and initializes it for a given device.
void ArrowDeviceArrayInit(struct ArrowDeviceArray* device_array,
                          struct ArrowDevice* device);

/// \brief Initialize an ArrowDeviceArrayView
///
/// Zeroes memory for the device array view struct. Callers must initialize the
/// array_view member using nanoarrow core functions that can initialize from
/// a type identifier or schema.
void ArrowDeviceArrayViewInit(struct ArrowDeviceArrayView* device_array_view);

/// \brief Release the underlying ArrowArrayView
void ArrowDeviceArrayViewReset(struct ArrowDeviceArrayView* device_array_view);

/// \brief Set ArrowArrayView buffer information from a device array
///
/// Whereas ArrowArrayViewSetArray() works ArrowArray objects with CPU-accessible memory,
/// it will crash arrays whose buffer addresses cannot be dereferenced.
ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error);

/// \brief Copy an ArrowDeviceArrayView to a device
ArrowErrorCode ArrowDeviceArrayViewCopy(struct ArrowDeviceArrayView* src,
                                        struct ArrowDevice* device_dst,
                                        struct ArrowDeviceArray* dst);

/// \brief Calculate if a copy is required to move an ArrowDeviceArrayView to a device
///
/// Returns 1 if a copy is required or 0 if the source array can be moved.
int ArrowDeviceArrayViewCopyRequired(struct ArrowDeviceArrayView* src,
                                     struct ArrowDevice* device_dst);

/// \brief Move an ArrowDeviceArrayView to a device if possible
///
/// Will attempt to zero-copy move a device array to the given device, falling back
/// to a copy otherwise.
ArrowErrorCode ArrowDeviceArrayTryMove(struct ArrowDeviceArray* src,
                                       struct ArrowDeviceArrayView* src_view,
                                       struct ArrowDevice* device_dst,
                                       struct ArrowDeviceArray* dst);

/// \brief Pointer to a statically-allocated CPU device singleton
struct ArrowDevice* ArrowDeviceCpu(void);

/// \brief Initialize a user-allocated device struct with a CPU device
void ArrowDeviceInitCpu(struct ArrowDevice* device);

/// \brief Resolve a device pointer from a type + identifier
///
/// Depending on which libraries this build of the device extension was built with,
/// some device types may or may not be supported. The CPU type is always supported.
/// Returns NULL for device that does not exist or cannot be returned as a singleton.
/// Callers must not release the pointed-to device.
struct ArrowDevice* ArrowDeviceResolve(ArrowDeviceType device_type, int64_t device_id);

ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                     struct ArrowDeviceBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst);

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst);

ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                     struct ArrowDeviceBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowDeviceBufferView dst);

/// \brief Initialize an ArrowDeviceArrayStream from an existing ArrowArrayStream
///
/// Wrap an ArrowArrayStream of ArrowDeviceArray objects already allocated by the
/// specified device as an ArrowDeviceArrayStream. This function moves the ownership of
/// array_stream to the device_array_stream. If this function returns NANOARROW_OK, the
/// caller is responsible for releasing the ArrowDeviceArrayStream.
ArrowErrorCode ArrowDeviceBasicArrayStreamInit(
    struct ArrowDeviceArrayStream* device_array_stream,
    struct ArrowArrayStream* array_stream, struct ArrowDevice* device);

/// @}

#ifdef __cplusplus
}
#endif

#endif
