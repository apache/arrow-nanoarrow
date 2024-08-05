
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

#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "nanoarrow/device/metal_impl.h"
#include "nanoarrow/nanoarrow_device.hpp"

// If non-null, caller must ->release() the return value. This doesn't
// release the underlying memory (which must be managed separately).
static MTL::Buffer* ArrowDeviceMetalWrapBufferNonOwning(MTL::Device* mtl_device,
                                                        const void* arbitrary_addr,
                                                        int64_t size_bytes = -1) {
  // Cache the page size from the system call
  static int pagesize = 0;
  if (pagesize == 0) {
    pagesize = getpagesize();
  }

  // If we don't know the size of the buffer yet, try pagesize
  if (size_bytes == -1) {
    size_bytes = pagesize;
  }

  // We can wrap any zero-size buffer
  if (size_bytes == 0) {
    return mtl_device->newBuffer(0, MTL::ResourceStorageModeShared);
  }

  int64_t allocation_size;
  if (size_bytes % pagesize == 0) {
    allocation_size = size_bytes;
  } else {
    allocation_size = (size_bytes / pagesize) + 1 * pagesize;
  }

  // Will return nullptr if the memory is improperly aligned
  return mtl_device->newBuffer(arbitrary_addr, allocation_size,
                               MTL::ResourceStorageModeShared, nullptr);
}

static uint8_t* ArrowDeviceMetalAllocatorReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  NANOARROW_UNUSED(allocator);

  // Cache the page size from the system call
  static int pagesize = 0;
  if (pagesize == 0) {
    pagesize = getpagesize();
  }

  int64_t allocation_size;
  if (new_size % pagesize == 0) {
    allocation_size = new_size;
  } else {
    allocation_size = (new_size / pagesize) + 1 * pagesize;
  }

  // If growing an existing buffer but the allocation size is still big enough,
  // return the same pointer and do nothing.
  if (ptr != nullptr && new_size >= old_size && new_size <= allocation_size) {
    return ptr;
  }

  int64_t copy_size;
  if (new_size > old_size) {
    copy_size = old_size;
  } else {
    copy_size = new_size;
  }

  void* new_ptr = nullptr;
  posix_memalign(&new_ptr, pagesize, allocation_size);
  if (new_ptr != nullptr && ptr != nullptr) {
    memcpy(new_ptr, ptr, copy_size);
  }

  if (ptr != nullptr) {
    free(ptr);
  }

  return reinterpret_cast<uint8_t*>(new_ptr);
}

static void ArrowDeviceMetalAllocatorFree(struct ArrowBufferAllocator* allocator,
                                          uint8_t* ptr, int64_t old_size) {
  NANOARROW_UNUSED(allocator);
  NANOARROW_UNUSED(old_size);
  free(ptr);
}

ArrowErrorCode ArrowDeviceMetalInitBuffer(struct ArrowBuffer* buffer) {
  buffer->allocator.reallocate = &ArrowDeviceMetalAllocatorReallocate;
  buffer->allocator.free = &ArrowDeviceMetalAllocatorFree;
  buffer->allocator.private_data = nullptr;
  buffer->data = nullptr;
  buffer->size_bytes = 0;
  buffer->capacity_bytes = 0;

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceMetalAlignArrayBuffers(struct ArrowArray* array) {
  struct ArrowBuffer* buffer;
  struct ArrowBuffer new_buffer;

  for (int64_t i = 0; i < array->n_buffers; i++) {
    buffer = ArrowArrayBuffer(array, i);
    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalInitBuffer(&new_buffer));
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppend(&new_buffer, buffer->data, buffer->size_bytes));
    ArrowBufferReset(buffer);
    ArrowBufferMove(&new_buffer, buffer);
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalAlignArrayBuffers(array->children[i]));
  }

  return NANOARROW_OK;
}

struct ArrowDeviceMetalArrayPrivate {
  struct ArrowArray parent;
  MTL::SharedEvent* event;
};

static void ArrowDeviceMetalArrayRelease(struct ArrowArray* array) {
  struct ArrowDeviceMetalArrayPrivate* private_data =
      (struct ArrowDeviceMetalArrayPrivate*)array->private_data;
  if (private_data->event != nullptr) {
    private_data->event->release();
  }
  ArrowArrayRelease(&private_data->parent);
  ArrowFree(private_data);
  array->release = nullptr;
}

static ArrowErrorCode ArrowDeviceMetalArrayInitAsync(
    struct ArrowDevice* device, struct ArrowDeviceArray* device_array,
    struct ArrowArray* array, void* sync_event, void* stream) {
  struct ArrowDeviceMetalArrayPrivate* private_data =
      (struct ArrowDeviceMetalArrayPrivate*)ArrowMalloc(
          sizeof(struct ArrowDeviceMetalArrayPrivate));
  if (private_data == nullptr) {
    return ENOMEM;
  }

  if (stream != NULL) {
    return EINVAL;
  }

  // One can create a new event with mtl_device->newSharedEvent();
  private_data->event = static_cast<MTL::SharedEvent*>(sync_event);

  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->array = *array;
  device_array->array.private_data = private_data;
  device_array->array.release = &ArrowDeviceMetalArrayRelease;
  ArrowArrayMove(array, &private_data->parent);

  device_array->device_id = device->device_id;
  device_array->device_type = device->device_type;
  device_array->sync_event = private_data->event;

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceMetalBufferInitAsync(struct ArrowDevice* device_src,
                                                      struct ArrowBufferView src,
                                                      struct ArrowDevice* device_dst,
                                                      struct ArrowBuffer* dst,
                                                      void* stream) {
  if (stream != nullptr) {
    return ENOTSUP;
  }

  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    struct ArrowBuffer tmp;
    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalInitBuffer(&tmp));
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&tmp, src.data.as_uint8, src.size_bytes));
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    struct ArrowBuffer tmp;
    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalInitBuffer(&tmp));
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&tmp, src.data.as_uint8, src.size_bytes));
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    struct ArrowBuffer tmp;
    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalInitBuffer(&tmp));
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&tmp, src.data.as_uint8, src.size_bytes));
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else {
    return ENOTSUP;
  }
}

static ArrowErrorCode ArrowDeviceMetalBufferMove(struct ArrowDevice* device_src,
                                                 struct ArrowBuffer* src,
                                                 struct ArrowDevice* device_dst,
                                                 struct ArrowBuffer* dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    // Check if the input is already aligned
    auto mtl_device = reinterpret_cast<MTL::Device*>(device_dst->private_data);
    MTL::Buffer* mtl_buffer =
        ArrowDeviceMetalWrapBufferNonOwning(mtl_device, src->data, src->size_bytes);
    if (mtl_buffer != nullptr) {
      mtl_buffer->release();
      ArrowBufferMove(src, dst);
      return NANOARROW_OK;
    } else {
      // Otherwise, return ENOTSUP to signal that a move is not possible
      return ENOTSUP;
    }
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    // Metal -> Metal is always just a move
    ArrowBufferMove(src, dst);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    // Metal -> CPU is also just a move since the memory is CPU accessible
    ArrowBufferMove(src, dst);
    return NANOARROW_OK;
  } else {
    return ENOTSUP;
  }
}

static ArrowErrorCode ArrowDeviceMetalBufferCopyAsync(struct ArrowDevice* device_src,
                                                      struct ArrowBufferView src,
                                                      struct ArrowDevice* device_dst,
                                                      struct ArrowBufferView dst,
                                                      void* stream) {
  if (stream != nullptr) {
    return ENOTSUP;
  }

  // This is all just memcpy since it's all living in the same address space
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    memcpy((void*)dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    memcpy((void*)dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy((void*)dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else {
    return ENOTSUP;
  }
}

static int ArrowDeviceMetalCopyRequiredCpuToMetal(MTL::Device* mtl_device,
                                                  struct ArrowArray* src) {
  // Only if all buffers in src can be wrapped as an MTL::Buffer
  for (int i = 0; i < src->n_buffers; i++) {
    MTL::Buffer* maybe_buffer =
        ArrowDeviceMetalWrapBufferNonOwning(mtl_device, src->buffers[i]);
    if (maybe_buffer == nullptr) {
      return true;
    }

    maybe_buffer->release();
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    int result = ArrowDeviceMetalCopyRequiredCpuToMetal(mtl_device, src->children[i]);
    if (result != 0) {
      return result;
    }
  }

  return false;
}

static ArrowErrorCode ArrowDeviceMetalSynchronize(struct ArrowDevice* device,
                                                  void* sync_event, void* stream,
                                                  struct ArrowError* error) {
  NANOARROW_UNUSED(device);
  NANOARROW_UNUSED(error);
  // TODO: sync events for Metal are harder than for CUDA
  // https://developer.apple.com/documentation/metal/resource_synchronization/synchronizing_events_between_a_gpu_and_the_cpu?language=objc
  // It would be much easier if sync_event were a command buffer

  // Something like:
  // auto listener = MTL::SharedEventListener::alloc();
  // listener->init();

  // auto event = reinterpret_cast<MTL::SharedEvent*>(sync_event);
  // event->notifyListener(
  //   listener, event->signaledValue(), ^(MTL::SharedEvent* event, uint64_t value) {
  //     event->signaledValue = value + 1;
  //   });

  // listener->release();

  // The case where we actually have to do something is not implemented
  if (sync_event != NULL || stream != NULL) {
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceMetalArrayMove(struct ArrowDevice* device_src,
                                                struct ArrowDeviceArray* src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowDeviceArray* dst) {
  // Note that the case where the devices are the same is handled before this

  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    // Check if we can do the move (i.e., if all buffers are page-aligned)
    auto mtl_device = reinterpret_cast<MTL::Device*>(device_dst->private_data);
    if (ArrowDeviceMetalCopyRequiredCpuToMetal(mtl_device, &src->array)) {
      return ENOTSUP;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceArrayInit(device_dst, dst, &src->array, src->sync_event));
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceMetalSynchronize(device_src, src->sync_event, nullptr, nullptr));
    ArrowDeviceArrayMove(src, dst);
    dst->device_type = device_dst->device_type;
    dst->device_id = device_dst->device_id;
    dst->sync_event = NULL;
    return NANOARROW_OK;
  }

  return ENOTSUP;
}

static void ArrowDeviceMetalRelease(struct ArrowDevice* device) {
  auto mtl_device = reinterpret_cast<MTL::Device*>(device->private_data);
  mtl_device->release();
  device->release = NULL;
}

struct ArrowDevice* ArrowDeviceMetalDefaultDevice(void) {
  static struct ArrowDevice* default_device_singleton = nullptr;
  if (default_device_singleton == nullptr) {
    default_device_singleton =
        (struct ArrowDevice*)ArrowMalloc(sizeof(struct ArrowDevice));
    int result = ArrowDeviceMetalInitDefaultDevice(default_device_singleton, nullptr);
    if (result != NANOARROW_OK) {
      ArrowFree(default_device_singleton);
      default_device_singleton = nullptr;
    }
  }

  return default_device_singleton;
}

ArrowErrorCode ArrowDeviceMetalInitDefaultDevice(struct ArrowDevice* device,
                                                 struct ArrowError* error) {
  MTL::Device* default_device = MTL::CreateSystemDefaultDevice();
  if (default_device == nullptr) {
    ArrowErrorSet(error, "No default device found");
    return EINVAL;
  }

  device->device_type = ARROW_DEVICE_METAL;
  device->device_id = static_cast<int64_t>(default_device->registryID());
  device->array_init = &ArrowDeviceMetalArrayInitAsync;
  device->array_move = &ArrowDeviceMetalArrayMove;
  device->buffer_init = &ArrowDeviceMetalBufferInitAsync;
  device->buffer_move = &ArrowDeviceMetalBufferMove;
  device->buffer_copy = &ArrowDeviceMetalBufferCopyAsync;
  device->synchronize_event = &ArrowDeviceMetalSynchronize;
  device->release = &ArrowDeviceMetalRelease;
  device->private_data = default_device;
  return NANOARROW_OK;
}
