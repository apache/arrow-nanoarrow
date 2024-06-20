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

#include "nanoarrow/nanoarrow.h"

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
#define ArrowDeviceArrayViewSetArrayMinimal \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewSetArrayMinimal)
#define ArrowDeviceArrayViewSetArrayAsync \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewSetArrayAsync)
#define ArrowDeviceArrayViewCopyAsync \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayViewCopyAsync)
#define ArrowDeviceArrayMoveToDevice \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceArrayMoveToDevice)
#define ArrowDeviceResolve NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceResolve)
#define ArrowDeviceCpu NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceCpu)
#define ArrowDeviceInitCpu NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceInitCpu)
#define ArrowDeviceBufferInitAsync \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferInitAsync)
#define ArrowDeviceBufferMove NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferMove)
#define ArrowDeviceBufferCopyAsync \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBufferCopyAsync)
#define ArrowDeviceBasicArrayStreamInit \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceBasicArrayStreamInit)

#define ArrowDeviceCuda NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceCuda)

#define ArrowDeviceMetalDefaultDevice \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceMetalDefaultDevice)
#define ArrowDeviceMetalInitDefaultDevice \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceMetalInitDefaultDevice)
#define ArrowDeviceMetalInitBuffer \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceMetalInitBuffer)
#define ArrowDeviceMetalAlignArrayBuffers \
  NANOARROW_SYMBOL(NANOARROW_NAMESPACE, ArrowDeviceMetalAlignArrayBuffers)

#endif

/// \defgroup nanoarrow_device Nanoarrow Device extension
///
/// Except where noted, objects are not thread-safe and clients should
/// take care to serialize accesses to methods.
///
/// @{

/// \brief Checks the nanoarrow runtime to make sure the run/build versions match
ArrowErrorCode ArrowDeviceCheckRuntime(struct ArrowError* error);

struct ArrowDeviceArrayView {
  struct ArrowDevice* device;
  struct ArrowArrayView array_view;
  void* sync_event;
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

  /// \brief Initialize an ArrowDeviceArray from a previously allocated ArrowArray
  ///
  /// Given a device and an uninitialized device_array, populate the fields of the
  /// device_array (including sync_event) appropriately. If NANOARROW_OK is returned,
  /// ownership of array is transferred to device_array. This function must allocate
  /// the appropriate sync_event and make its address available as
  /// device_array->sync_event (if sync_event applies to this device type).
  ArrowErrorCode (*array_init)(struct ArrowDevice* device,
                               struct ArrowDeviceArray* device_array,
                               struct ArrowArray* array, void* sync_event);

  /// \brief Move an ArrowDeviceArray between devices without copying buffers
  ///
  /// Some devices can move an ArrowDeviceArray without an explicit buffer copy,
  /// although the performance characteristics of the moved array may be different
  /// than that of an explicitly copied one depending on the device.
  ArrowErrorCode (*array_move)(struct ArrowDevice* device_src,
                               struct ArrowDeviceArray* src,
                               struct ArrowDevice* device_dst,
                               struct ArrowDeviceArray* dst);

  /// \brief Initialize an owning buffer from existing content
  ///
  /// Creates a new buffer whose data member can be accessed by the GPU by
  /// copying existing content.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_init)(struct ArrowDevice* device_src,
                                struct ArrowBufferView src,
                                struct ArrowDevice* device_dst, struct ArrowBuffer* dst,
                                void* stream);

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
                                struct ArrowBufferView src,
                                struct ArrowDevice* device_dst,
                                struct ArrowBufferView dst, void* stream);

  /// \brief Wait for an event on the CPU host
  ArrowErrorCode (*synchronize_event)(struct ArrowDevice* device, void* sync_event,
                                      void* stream, struct ArrowError* error);

  /// \brief Release this device and any resources it holds
  void (*release)(struct ArrowDevice* device);

  /// \brief Opaque, implementation-specific data.
  void* private_data;
};

/// \brief Initialize an ArrowDeviceArray
///
/// Given an ArrowArray whose buffers/release callback has been set appropriately,
/// initialize an ArrowDeviceArray.
ArrowErrorCode ArrowDeviceArrayInit(struct ArrowDevice* device,
                                    struct ArrowDeviceArray* device_array,
                                    struct ArrowArray* array, void* sync_event);

/// \brief Initialize an ArrowDeviceArrayView
///
/// Zeroes memory for the device array view struct. Callers must initialize the
/// array_view member using nanoarrow core functions that can initialize from
/// a type identifier or schema.
void ArrowDeviceArrayViewInit(struct ArrowDeviceArrayView* device_array_view);

/// \brief Release the underlying ArrowArrayView
void ArrowDeviceArrayViewReset(struct ArrowDeviceArrayView* device_array_view);

/// \brief Set minimal ArrowArrayView buffer information from a device array
///
/// A thin wrapper around ArrowArrayViewSetArrayMinimal() that does not attempt
/// to resolve buffer sizes of variable-length buffers by copying data from the device.
ArrowErrorCode ArrowDeviceArrayViewSetArrayMinimal(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error);

/// \brief Set ArrowArrayView buffer information from a device array
///
/// Runs ArrowDeviceArrayViewSetArrayMinimal() but also sets buffer sizes for
/// variable-length buffers by copying data from the device. This function will block on
/// the device_array's sync_event.
static inline ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error);

ArrowErrorCode ArrowDeviceArrayViewSetArrayAsync(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    void* stream, struct ArrowError* error);

/// \brief Copy an ArrowDeviceArrayView to a device
ArrowErrorCode ArrowDeviceArrayViewCopyAsync(struct ArrowDeviceArrayView* src,
                                             struct ArrowDevice* device_dst,
                                             struct ArrowDeviceArray* dst, void* stream);

static inline ArrowErrorCode ArrowDeviceArrayViewCopy(struct ArrowDeviceArrayView* src,
                                                      struct ArrowDevice* device_dst,
                                                      struct ArrowDeviceArray* dst);

/// \brief Move an ArrowDeviceArray to a device if possible
///
/// Will attempt to move a device array to a device without copying buffers.
/// This may result in a device array with different performance charateristics
/// than an array that was copied.
ArrowErrorCode ArrowDeviceArrayMoveToDevice(struct ArrowDeviceArray* src,
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

ArrowErrorCode ArrowDeviceBufferInitAsync(struct ArrowDevice* device_src,
                                          struct ArrowBufferView src,
                                          struct ArrowDevice* device_dst,
                                          struct ArrowBuffer* dst, void* stream);

static inline ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                                   struct ArrowBufferView src,
                                                   struct ArrowDevice* device_dst,
                                                   struct ArrowBuffer* dst);

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst);

static inline ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                                   struct ArrowBufferView src,
                                                   struct ArrowDevice* device_dst,
                                                   struct ArrowBufferView dst);

ArrowErrorCode ArrowDeviceBufferCopyAsync(struct ArrowDevice* device_src,
                                          struct ArrowBufferView src,
                                          struct ArrowDevice* device_dst,
                                          struct ArrowBufferView dst, void* stream);

/// \brief Initialize an ArrowDeviceArrayStream from an existing ArrowArrayStream
///
/// Wrap an ArrowArrayStream of ArrowDeviceArray objects already allocated by the
/// specified device as an ArrowDeviceArrayStream. This function moves the ownership
/// of array_stream to the device_array_stream. If this function returns NANOARROW_OK,
/// the caller is responsible for releasing the ArrowDeviceArrayStream.
ArrowErrorCode ArrowDeviceBasicArrayStreamInit(
    struct ArrowDeviceArrayStream* device_array_stream,
    struct ArrowArrayStream* array_stream, struct ArrowDevice* device);

/// @}

/// \defgroup nanoarrow_device_cuda CUDA Device extension
///
/// A CUDA (i.e., `cuda_runtime_api.h`) implementation of the Arrow C Device
/// interface.
///
/// @{

/// \brief Get a CUDA device from type and ID
///
/// device_type must be one of ARROW_DEVICE_CUDA or ARROW_DEVICE_CUDA_HOST;
/// device_id must be between 0 and cudaGetDeviceCount - 1.
struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id);

/// @}

/// \defgroup nanoarrow_device_metal Apple Metal Device extension
///
/// An Apple Metal implementation of the Arrow C Device interface, primarily targeted to
/// the M1 series of CPU/GPUs that feature shared CPU/GPU memory. Even though the memory
/// regions are shared, it is currently not possible to wrap an arbitrary CPU memory
/// region as an `MTL::Buffer*` unless that memory region is page-aligned. Because of
/// this, a copy is still required in most cases to make memory GPU accessible. After GPU
/// calculations are complete; however, moving the buffers back to the CPU is zero-copy.
///
/// Sync events are represented as an `MTL::Event*`. The degree to which the pointers
/// to `MTL::Event*` are stable across metal-cpp versions/builds is currently unknown.
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
ArrowErrorCode ArrowDeviceMetalInitBuffer(struct ArrowBuffer* buffer);

/// \brief Convert an ArrowArray to buffers that use the Metal allocator
///
/// Replaces buffers from a given ArrowArray with ones that use the Metal
/// allocator, copying existing content where necessary. The array is still
/// valid to use just like a normal ArrowArray that was initialized with
/// ArrowArrayInitFromType() (i.e., it can be appended to and finished with
/// validation).
ArrowErrorCode ArrowDeviceMetalAlignArrayBuffers(struct ArrowArray* array);

/// @}
// Inline implementations

static inline ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                                   struct ArrowBufferView src,
                                                   struct ArrowDevice* device_dst,
                                                   struct ArrowBufferView dst) {
  return ArrowDeviceBufferCopyAsync(device_src, src, device_dst, dst, NULL);
}

static inline ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                                   struct ArrowBufferView src,
                                                   struct ArrowDevice* device_dst,
                                                   struct ArrowBuffer* dst) {
  return ArrowDeviceBufferInitAsync(device_src, src, device_dst, dst, NULL);
}

static inline ArrowErrorCode ArrowDeviceArrayViewCopy(struct ArrowDeviceArrayView* src,
                                                      struct ArrowDevice* device_dst,
                                                      struct ArrowDeviceArray* dst) {
  return ArrowDeviceArrayViewCopyAsync(src, device_dst, dst, NULL);
}

static inline ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error) {
  return ArrowDeviceArrayViewSetArrayAsync(device_array_view, device_array, NULL, error);
}

#ifdef __cplusplus
}
#endif

#endif
