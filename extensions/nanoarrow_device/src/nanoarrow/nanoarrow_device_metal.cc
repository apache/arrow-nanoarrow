
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

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>

#include "nanoarrow_device.hpp"

#include "nanoarrow_device_metal.h"

// If non-null, caller must ->release() the return value. This doesn't
// release the underlying memory (which must be managed separately).
static MTL::Buffer* ArrowDeviceMetalWrapBufferNonOwning(MTL::Device* mtl_device,
                                                        const void* arbitrary_addr,
                                                        int64_t size_bytes) {
  // We can wrap any zero-size buffer
  if (size_bytes == 0) {
    return mtl_device->newBuffer(0, MTL::ResourceStorageModeShared);
  }

  // Cache the page size from the system call
  static int pagesize = 0;
  if (pagesize == 0) {
    pagesize = getpagesize();
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
  free(ptr);
}

void ArrowDeviceMetalInitBuffer(struct ArrowBuffer* buffer) {
  buffer->allocator.reallocate = &ArrowDeviceMetalAllocatorReallocate;
  buffer->allocator.free = &ArrowDeviceMetalAllocatorFree;
  buffer->allocator.private_data = nullptr;
  buffer->data = nullptr;
  buffer->size_bytes = 0;
  buffer->capacity_bytes = 0;
}

ArrowErrorCode ArrowDeviceMetalAlignArrayBuffers(struct ArrowArray* array) {
  struct ArrowBuffer* buffer;
  struct ArrowBuffer new_buffer;

  for (int64_t i = 0; i < array->n_buffers; i++) {
    buffer = ArrowArrayBuffer(array, i);
    ArrowDeviceMetalInitBuffer(&new_buffer);
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

static ArrowErrorCode ArrowDeviceMetalBufferInit(struct ArrowDevice* device_src,
                                                 struct ArrowBufferView src,
                                                 struct ArrowDevice* device_dst,
                                                 struct ArrowBuffer* dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    struct ArrowBuffer tmp;
    ArrowDeviceMetalInitBuffer(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(
        &tmp, src.data.as_uint8,
        src.size_bytes));
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    struct ArrowBuffer tmp;
    ArrowDeviceMetalInitBuffer(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(
        &tmp, src.data.as_uint8,
        src.size_bytes));
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    struct ArrowBuffer tmp;
    ArrowDeviceMetalInitBuffer(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(
        &tmp, src.data.as_uint8,
        src.size_bytes));
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
    }

    // Otherwise, initialize a new buffer and copy
    struct ArrowBuffer tmp;
    ArrowDeviceMetalInitBuffer(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(&tmp, src->data, src->size_bytes));
    ArrowBufferMove(&tmp, dst);
    ArrowBufferReset(src);
    return NANOARROW_OK;
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

static ArrowErrorCode ArrowDeviceMetalBufferCopy(struct ArrowDevice* device_src,
                                                 struct ArrowBufferView src,
                                                 struct ArrowDevice* device_dst,
                                                 struct ArrowBufferView dst) {
  // This is all just memcpy since it's all living in the same address space
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    memcpy((void*)dst.data.as_uint8,
           src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    memcpy((void*)dst.data.as_uint8,
           src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy((void*)dst.data.as_uint8,
           src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else {
    return ENOTSUP;
  }
}

static int ArrowDeviceMetalCopyRequired(struct ArrowDevice* device_src,
                                        struct ArrowArrayView* src,
                                        struct ArrowDevice* device_dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_METAL) {
    // Only if all buffers in src can be wrapped as an MTL::Buffer
    auto mtl_device = reinterpret_cast<MTL::Device*>(device_dst->private_data);
    for (int i = 0; i < 3; i++) {
      if (src->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
        break;
      }

      MTL::Buffer* maybe_buffer = ArrowDeviceMetalWrapBufferNonOwning(
          mtl_device, src->buffer_views[i].data.data, src->buffer_views[i].size_bytes);
      if (maybe_buffer == nullptr) {
        return true;
      }

      maybe_buffer->release();
    }

    for (int64_t i = 0; i < src->n_children; i++) {
      int result = ArrowDeviceMetalCopyRequired(device_src, src->children[i], device_dst);
      if (result != 0) {
        return result;
      }
    }

    return false;

  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_METAL) {
    // Metal -> Metal is always a move
    return 0;
  } else if (device_src->device_type == ARROW_DEVICE_METAL &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    // We can always go from super-aligned metal land to CPU land
    return 0;
  } else {
    // Fall back to the other device's implementation
    return -1;
  }
}

static ArrowErrorCode ArrowDeviceMetalSynchronize(struct ArrowDevice* device,
                                                  void* sync_event,
                                                  struct ArrowError* error) {
  if (sync_event == nullptr) {
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
  device->array_init = nullptr;
  device->array_move = nullptr;
  device->buffer_init = &ArrowDeviceMetalBufferInit;
  device->buffer_move = &ArrowDeviceMetalBufferMove;
  device->buffer_copy = &ArrowDeviceMetalBufferCopy;
  device->copy_required = &ArrowDeviceMetalCopyRequired;
  device->synchronize_event = &ArrowDeviceMetalSynchronize;
  device->release = &ArrowDeviceMetalRelease;
  device->private_data = default_device;
  return NANOARROW_OK;
}
