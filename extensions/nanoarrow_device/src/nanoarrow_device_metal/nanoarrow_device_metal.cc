
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

ArrowErrorCode ArrowDeviceInitMetalDefault(struct ArrowDevice* device,
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
