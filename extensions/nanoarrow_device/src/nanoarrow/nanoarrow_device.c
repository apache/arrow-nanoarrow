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

#include "nanoarrow.h"

#include "nanoarrow_device.h"

ArrowErrorCode ArrowDeviceCheckRuntime(struct ArrowError* error) {
  const char* nanoarrow_runtime_version = ArrowNanoarrowVersion();
  const char* nanoarrow_ipc_build_time_version = NANOARROW_VERSION;

  if (strcmp(nanoarrow_runtime_version, nanoarrow_ipc_build_time_version) != 0) {
    ArrowErrorSet(error, "Expected nanoarrow runtime version '%s' but found version '%s'",
                  nanoarrow_ipc_build_time_version, nanoarrow_runtime_version);
    return EINVAL;
  }

  return NANOARROW_OK;
}

void ArrowDeviceArrayInit(struct ArrowDeviceArray* device_array,
                          struct ArrowDevice* device) {
  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->device_type = device->device_type;
  device_array->device_id = device->device_id;
}

static ArrowErrorCode ArrowDeviceCpuCopyTo(struct ArrowDevice* device,
                                           struct ArrowBufferView src,
                                           struct ArrowDevice* device_dst,
                                           struct ArrowBuffer* dst, void** sync_event,
                                           struct ArrowError* error) {
  switch (device->device_type) {
    case ARROW_DEVICE_CPU:
      ArrowBufferInit(dst);
      dst->allocator = device->allocator;
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppendBufferView(dst, src), error);
      *sync_event = NULL;
      return NANOARROW_OK;
    default:
      return device_dst->copy_from(device_dst, dst, device, src, sync_event, error);
  }
}

static ArrowErrorCode ArrowDeviceCpuCopyFrom(
    struct ArrowDevice* device, struct ArrowBuffer* dst, struct ArrowDevice* device_src,
    struct ArrowBufferView src, void** sync_event, struct ArrowError* error) {
  switch (device->device_type) {
    case ARROW_DEVICE_CPU:
      ArrowBufferInit(dst);
      dst->allocator = device->allocator;
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppendBufferView(dst, src), error);
      *sync_event = NULL;
      return NANOARROW_OK;
    default:
      return device_src->copy_to(device_src, src, device, dst, sync_event, error);
  }
}

static ArrowErrorCode ArrowDeviceCpuSynchronize(struct ArrowDevice* device,
                                                struct ArrowDevice* device_event,
                                                void* sync_event,
                                                struct ArrowError* error) {
  switch (device_event->device_type) {
    case ARROW_DEVICE_CPU:
      if (sync_event != NULL) {
        return EINVAL;
      } else {
        return NANOARROW_OK;
      }
    default:
      return device_event->synchronize_event(device_event, device, sync_event, error);
  }
}

struct ArrowDevice* ArrowDeviceCpu(void) {
  static struct ArrowDevice* cpu_device_singleton = NULL;
  if (cpu_device_singleton == NULL) {
    cpu_device_singleton = (struct ArrowDevice*)ArrowMalloc(sizeof(struct ArrowDevice));
    cpu_device_singleton->device_type = ARROW_DEVICE_CPU;
    cpu_device_singleton->device_id = 0;
    cpu_device_singleton->allocator = ArrowBufferAllocatorDefault();
    cpu_device_singleton->copy_from = &ArrowDeviceCpuCopyFrom;
    cpu_device_singleton->copy_to = &ArrowDeviceCpuCopyTo;
    cpu_device_singleton->synchronize_event = &ArrowDeviceCpuSynchronize;
    cpu_device_singleton->private_data = NULL;
  }

  return cpu_device_singleton;
}

ArrowErrorCode ArrowBasicDeviceArrayStreamInit(
    struct ArrowDeviceArrayStream* device_array_stream,
    struct ArrowArrayStream* array_stream) {
  return ENOTSUP;
}
