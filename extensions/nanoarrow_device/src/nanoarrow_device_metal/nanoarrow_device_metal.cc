
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

#include <Metal/Metal.hpp>

#include "nanoarrow_device.hpp"

#include "nanoarrow_device_metal.h"

// Wrap reference-counted NS objects
template <typename T>
class Owner {
 public:
  Owner() : ptr_(nullptr) {}
  Owner(T* ptr) : ptr_(ptr) {}

  T* operator->() { return ptr_; }

  void reset(T* ptr = nullptr) {
    ptr_->release();
    ptr_ = ptr;
  }

  T* get() { return ptr_; }

  ~Owner() { reset(); }

 private:
  T* ptr_;
};

static uint8_t* ArrowDeviceMetalAllocatorReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  auto mtl_buffer = reinterpret_cast<MTL::Buffer*>(allocator->private_data);

  // After a failed allocation, this allocator currently can't reallocate
  if (mtl_buffer == nullptr) {
    return nullptr;
  }

  if (new_size < 64) {
    new_size = 64;
  }

  old_size = mtl_buffer->length();
  int64_t copy_size = new_size;
  if (new_size == old_size) {
    return reinterpret_cast<uint8_t*>(mtl_buffer->contents());
  } else if (new_size > old_size) {
    copy_size = old_size;
  }

  auto mtl_buffer_new =
      mtl_buffer->device()->newBuffer(new_size, MTL::ResourceStorageModeShared);

  // It's slightly simpler to not support reallocating after a failed allocation,
  // although that can probably be fixed in the buffer logic that doesn't call
  // allocator->free() on NULL.
  if (mtl_buffer_new == nullptr) {
    mtl_buffer->release();
    allocator->private_data = nullptr;
    return nullptr;
  }

  memcpy(mtl_buffer_new->contents(), mtl_buffer->contents(), copy_size);
  mtl_buffer->release();
  allocator->private_data = mtl_buffer_new;
  return reinterpret_cast<uint8_t*>(mtl_buffer_new->contents());
}

static void ArrowDeviceMetalAllocatorFree(struct ArrowBufferAllocator* allocator,
                                          uint8_t* ptr, int64_t old_size) {
  auto mtl_buffer = reinterpret_cast<MTL::Buffer*>(allocator->private_data);
  if (mtl_buffer != nullptr) {
    mtl_buffer->release();
  }
}

ArrowErrorCode ArrowDeviceMetalInitCpuBuffer(struct ArrowDevice* device,
                                             struct ArrowBuffer* buffer,
                                             struct ArrowBufferView initial_content) {
  if (device->device_type != ARROW_DEVICE_METAL || device->release == nullptr) {
    return EINVAL;
  }

  auto mtl_device = reinterpret_cast<MTL::Device*>(device->private_data);
  MTL::Buffer* mtl_buffer;
  if (initial_content.size_bytes > 0) {
    mtl_buffer =
        mtl_device->newBuffer(initial_content.data.data, initial_content.size_bytes,
                              MTL::ResourceStorageModeShared);
  } else {
    mtl_buffer = mtl_device->newBuffer(64, MTL::ResourceStorageModeShared);
  }

  if (mtl_buffer == nullptr) {
    return ENOMEM;
  }

  buffer->allocator.reallocate = &ArrowDeviceMetalAllocatorReallocate;
  buffer->allocator.free = &ArrowDeviceMetalAllocatorFree;
  buffer->allocator.private_data = mtl_buffer;
  buffer->data = reinterpret_cast<uint8_t*>(mtl_buffer->contents());
  buffer->size_bytes = 0;
  buffer->capacity_bytes = mtl_buffer->length();
  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceMetalInitCpuArrayBuffers(struct ArrowDevice* device,
                                                   struct ArrowArray* array) {
  struct ArrowBuffer* buffer;
  struct ArrowBufferView contents;
  struct ArrowBuffer new_buffer;

  for (int64_t i = 0; i < array->n_buffers; i++) {
    buffer = ArrowArrayBuffer(array, i);
    contents.data.data = buffer->data;
    contents.size_bytes = buffer->size_bytes;

    NANOARROW_RETURN_NOT_OK(ArrowDeviceMetalInitCpuBuffer(device, &new_buffer, contents));
    ArrowBufferReset(buffer);
    ArrowBufferMove(&new_buffer, buffer);
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceMetalInitCpuArrayBuffers(device, array->children[i]));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceMetalCopyBuffer(struct ArrowDevice* device_src,
                                                 struct ArrowBufferView src,
                                                 struct ArrowDevice* device_dst,
                                                 struct ArrowBuffer* dst,
                                                 void** sync_event) {
  return ENOTSUP;
}

static ArrowErrorCode ArrowDeviceMetalSynchronize(struct ArrowDevice* device,
                                                  struct ArrowDevice* device_event,
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
  device->copy_buffer = &ArrowDeviceMetalCopyBuffer;
  device->synchronize_event = &ArrowDeviceMetalSynchronize;
  device->release = &ArrowDeviceMetalRelease;
  device->private_data = default_device;
  return NANOARROW_OK;
}
