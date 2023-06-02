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
/// Arrow Columnar Format specification
/// (https://arrow.apache.org/docs/format/Columnar.html). See the Arrow documentation for
/// detailed documentation of these structures.
///
/// @{

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

/// \defgroup arrow-device-types Device Types
/// These macros are compatible with the dlpack DLDeviceType values,
/// using the same value for each enum as the equivalent kDL<type>
/// from dlpack.h. This list should continue to be kept in sync with
/// the equivalent dlpack.h enum values over time to ensure
/// compatibility, rather than potentially diverging.
///
/// To ensure predictability with the ABI we use macros instead of
/// an enum so the storage type is not compiler dependent.
///
/// @{

/// \brief DeviceType for the allocated memory
typedef int32_t ArrowDeviceType;

/// \brief CPU device, same as using ArrowArray directly
#define ARROW_DEVICE_CPU 1
/// \brief CUDA GPU Device
#define ARROW_DEVICE_CUDA 2
/// \brief Pinned CUDA CPU memory by cudaMallocHost
#define ARROW_DEVICE_CUDA_HOST 3
/// \brief OpenCL Device
#define ARROW_DEVICE_OPENCL 4
/// \brief Vulkan buffer for next-gen graphics
#define ARROW_DEVICE_VULKAN 7
/// \brief Metal for Apple GPU
#define ARROW_DEVICE_METAL 8
/// \brief Verilog simulator buffer
#define ARROW_DEVICE_VPI 9
/// \brief ROCm GPUs for AMD GPUs
#define ARROW_DEVICE_ROCM 10
/// \brief Pinned ROCm CPU memory allocated by hipMallocHost
#define ARROW_DEVICE_ROCM_HOST 11
/// \brief Reserved for extension
///
/// used to quickly test extension devices, semantics
/// can differ based on the implementation
#define ARROW_DEVICE_EXT_DEV 12
/// \brief CUDA managed/unified memory allocated by cudaMallocManaged
#define ARROW_DEVICE_CUDA_MANAGED 13
/// \brief unified shared memory allocated on a oneAPI
/// non-partitioned device.
///
/// A call to the oneAPI runtime is required to determine the device
/// type, the USM allocation type, and the sycl context it is bound to.
#define ARROW_DEVICE_ONEAPI 14
/// \brief GPU support for next-gen WebGPU standard
#define ARROW_DEVICE_WEBGPU 15
/// \brief Qualcomm Hexagon DSP
#define ARROW_DEVICE_HEXAGON 16

/// @}

/// \brief Struct for passing an Arrow Array alongside
/// device memory information.
struct ArrowDeviceArray {
  /// \brief the Allocated Array
  ///
  /// the buffers in the array (along with the buffers of any
  /// children) are what is allocated on the device.
  ///
  /// the private_data and release callback of the arrow array
  /// should contain any necessary information and structures
  /// related to freeing the array according to the device it
  /// is allocated on, rather than having a separate release
  /// callback embedded here.
  struct ArrowArray array;
  /// \brief The device id to identify a specific device
  /// if multiple of this type are on the system.
  ///
  /// the semantics of the id will be hardware dependant.
  int64_t device_id;
  /// \brief The type of device which can access this memory.
  ArrowDeviceType device_type;
  /// \brief An event-like object to synchronize on if needed.
  ///
  /// Many devices, like GPUs, are primarily asynchronous with
  /// respect to CPU processing. As such in order to safely access
  /// memory, it is often necessary to have an object to synchronize
  /// processing on. Since different devices will use different types
  /// to specify this we use a void* that can be coerced into
  /// whatever the device appropriate type is (e.g. cudaEvent_t for
  /// CUDA and hipEvent_t for HIP).
  ///
  /// If synchronization is not needed this can be null. If this is
  /// non-null, then it should be used to call the appropriate sync
  /// method for the device (e.g. cudaStreamWaitEvent / hipStreamWaitEvent).
  ///
  /// Expected type to coerce this void* to depending on device type:
  ///   cuda: cudaEvent_t*
  ///   ROCm: hipEvent_t*
  ///   OpenCL: cl_event*
  ///   Vulkan: VkEvent*
  ///   Metal: MTLEvent*
  ///   OneAPI: sycl::event*
  ///
  void* sync_event;
  /// \brief Reserved bytes for future expansion.
  ///
  /// As non-CPU development expands we can update this struct
  /// without ABI breaking changes. This also rounds out the
  /// total size of this struct to be 128 bytes (power of 2)
  /// on 64-bit systems. These bytes should be zero'd out after
  /// allocation in order to ensure safe evolution of the ABI in
  /// the future.
  int64_t reserved[3];
};

#endif  // ARROW_C_DEVICE_DATA_INTERFACE

#ifndef ARROW_C_DEVICE_STREAM_INTERFACE
#define ARROW_C_DEVICE_STREAM_INTERFACE

/// \brief Equivalent to ArrowArrayStream, but for ArrowDeviceArrays.
///
/// This stream is intended to provide a stream of data on a single
/// device, if a producer wants data to be produced on multiple devices
/// then multiple streams should be provided. One per device.
struct ArrowDeviceArrayStream {
  /// \brief The device that this stream produces data on.
  ///
  /// All ArrowDeviceArrays that are produced by this
  /// stream should have the same device_type as set
  /// here. Including it here in the stream object is
  /// a convenience to allow consumers simpler processing
  /// since they can assume all arrays that result from
  /// this stream to be on this device type.
  ArrowDeviceType device_type;

  /// \brief Callback to get the stream schema
  /// (will be the same for all arrays in the stream).
  ///
  /// If successful, the ArrowSchema must be released independantly from the stream.
  /// The schema should be accessible via CPU memory.
  ///
  /// \param[in] self The ArrowDeviceArrayStream object itself
  /// \param[out] out C struct to export the schema to
  /// \return 0 if successful, an `errno`-compatible error code otherwise.
  int (*get_schema)(struct ArrowDeviceArrayStream* self, struct ArrowSchema* out);

  /// \brief Callback to get the next array
  ///
  /// If there is no error and the returned array has been released, the stream
  /// has ended. If successful, the ArrowArray must be released independently
  /// from the stream.
  ///
  /// \param[in] self The ArrowDeviceArrayStream object itself
  /// \param[out] out C struct where to export the Array and device info
  /// \return 0 if successful, an `errno`-compatible error code otherwise.
  int (*get_next)(struct ArrowDeviceArrayStream* self, struct ArrowDeviceArray* out);

  /// \brief Callback to get optional detailed error information.
  ///
  /// This must only be called if the last stream operation failed
  /// with a non-0 return code.
  ///
  /// The returned pointer is only valid until the next operation on this stream
  /// (including release).
  ///
  /// \param[in] self The ArrowDeviceArrayStream object itself
  /// \return pointer to a null-terminated character array describing
  /// the last error, or NULL if no description is available.
  const char* (*get_last_error)(struct ArrowDeviceArrayStream* self);

  /// \brief Release callback: release the stream's own resources.
  ///
  /// Note that arrays returned by `get_next` must be individually released.
  ///
  /// \param[in] self The ArrowDeviceArrayStream object itself
  void (*release)(struct ArrowDeviceArrayStream* self);

  /// \brief Opaque producer-specific data
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
  /// \brief Device-defined handle for a buffer. For the CPU device, this is
  /// a normal memory address.
  void* private_data;

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
                                struct ArrowDevice* device_dst, struct ArrowBuffer* dst,
                                void** sync_event);

  /// \brief Move an owning buffer to a device
  ///
  /// Creates a new buffer whose data member can be accessed by the GPU by
  /// moving an existing buffer. If NANOARROW_OK is returned, src will have
  /// been released or moved by the implementation and dst must be released by
  /// the caller.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_move)(struct ArrowDevice* device_src, struct ArrowBuffer* src,
                                struct ArrowDevice* device_dst, struct ArrowBuffer* dst,
                                void** sync_event);

  /// \brief Copy a section of memory into a preallocated buffer
  ///
  /// As opposed to the other buffer operations, this is designed to support
  /// copying very small slices of memory.
  /// Implementations must check device_src and device_dst and return ENOTSUP if
  /// not prepared to handle this operation.
  ArrowErrorCode (*buffer_copy)(struct ArrowDevice* device_src,
                                struct ArrowDeviceBufferView src,
                                struct ArrowDevice* device_dst,
                                struct ArrowDeviceBufferView dst, void** sync_event);

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

/// \brief Set ArrowArrayView buffer information from a device array
///
/// Whereas ArrowArrayViewSetArray() works ArrowArray objects with CPU-accessible memory,
/// it will crash arrays whose buffer addresses cannot be dereferenced.
ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error);

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
                                     struct ArrowBuffer* dst, void** sync_event);

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst, void** sync_event);

ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                     struct ArrowDeviceBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowDeviceBufferView dst, void** sync_event);

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
