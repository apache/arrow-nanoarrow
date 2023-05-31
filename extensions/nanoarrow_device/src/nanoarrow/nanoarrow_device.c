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

static ArrowErrorCode ArrowDeviceCpuBufferInit(struct ArrowDevice* device_src,
                                               struct ArrowBufferView src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBuffer* dst,
                                               void** sync_event) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferInit(dst);
  dst->allocator = ArrowBufferAllocatorDefault();
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendBufferView(dst, src));
  *sync_event = NULL;
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferMove(struct ArrowDevice* device_src,
                                               struct ArrowBuffer* src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBuffer* dst,
                                               void** sync_event) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferMove(src, dst);
  *sync_event = NULL;
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferCopy(struct ArrowDevice* device_src,
                                               struct ArrowBufferView src,
                                               struct ArrowDevice* device_dst,
                                               uint8_t* dst, void** sync_event) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  memcpy(dst, src.data.data, src.size_bytes);
  *sync_event = NULL;
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuSynchronize(struct ArrowDevice* device,
                                                struct ArrowDevice* device_event,
                                                void* sync_event,
                                                struct ArrowError* error) {
  switch (device_event->device_type) {
    case ARROW_DEVICE_CPU:
      if (sync_event != NULL) {
        ArrowErrorSet(error, "Expected NULL sync_event for ARROW_DEVICE_CPU but got %p",
                      sync_event);
        return EINVAL;
      } else {
        return NANOARROW_OK;
      }
    default:
      return device_event->synchronize_event(device_event, device, sync_event, error);
  }
}

static void ArrowDeviceCpuRelease(struct ArrowDevice* device) { device->release = NULL; }

struct ArrowDevice* ArrowDeviceCpu(void) {
  static struct ArrowDevice* cpu_device_singleton = NULL;
  if (cpu_device_singleton == NULL) {
    cpu_device_singleton = (struct ArrowDevice*)ArrowMalloc(sizeof(struct ArrowDevice));
    ArrowDeviceInitCpu(cpu_device_singleton);
  }

  return cpu_device_singleton;
}

void ArrowDeviceInitCpu(struct ArrowDevice* device) {
  device->device_type = ARROW_DEVICE_CPU;
  device->device_id = 0;
  device->buffer_init = &ArrowDeviceCpuBufferInit;
  device->buffer_move = &ArrowDeviceCpuBufferMove;
  device->buffer_copy = &ArrowDeviceCpuBufferCopy;
  device->synchronize_event = &ArrowDeviceCpuSynchronize;
  device->release = &ArrowDeviceCpuRelease;
  device->private_data = NULL;
}

ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst, void** sync_event) {
  int result = device_dst->buffer_init(device_src, src, device_dst, dst, sync_event);
  if (result == ENOTSUP) {
    result = device_src->buffer_init(device_src, src, device_dst, dst, sync_event);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst, void** sync_event) {
  int result = device_dst->buffer_move(device_src, src, device_dst, dst, sync_event);
  if (result == ENOTSUP) {
    result = device_src->buffer_move(device_src, src, device_dst, dst, sync_event);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst, uint8_t* dst,
                                     void** sync_event) {
  int result = device_dst->buffer_copy(device_src, src, device_dst, dst, sync_event);
  if (result == ENOTSUP) {
    result = device_src->buffer_copy(device_src, src, device_dst, dst, sync_event);
  }

  return result;
}

struct ArrowBasicDeviceArrayStreamPrivate {
  struct ArrowDevice* device;
  struct ArrowArrayStream naive_stream;
};

static int ArrowDeviceBasicArrayStreamGetSchema(
    struct ArrowDeviceArrayStream* array_stream, struct ArrowSchema* schema) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  return private_data->naive_stream.get_schema(&private_data->naive_stream, schema);
}

static int ArrowDeviceBasicArrayStreamGetNext(struct ArrowDeviceArrayStream* array_stream,
                                              struct ArrowDeviceArray* device_array) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(
      private_data->naive_stream.get_next(&private_data->naive_stream, &tmp));
  ArrowDeviceArrayInit(device_array, private_data->device);
  ArrowArrayMove(&tmp, &device_array->array);
  return NANOARROW_OK;
}

static const char* ArrowDeviceBasicArrayStreamGetLastError(
    struct ArrowDeviceArrayStream* array_stream) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  return private_data->naive_stream.get_last_error(&private_data->naive_stream);
}

static void ArrowDeviceBasicArrayStreamRelease(
    struct ArrowDeviceArrayStream* array_stream) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)array_stream->private_data;
  private_data->naive_stream.release(&private_data->naive_stream);
  ArrowFree(private_data);
  array_stream->release = NULL;
}

ArrowErrorCode ArrowDeviceBasicArrayStreamInit(
    struct ArrowDeviceArrayStream* device_array_stream,
    struct ArrowArrayStream* array_stream, struct ArrowDevice* device) {
  struct ArrowBasicDeviceArrayStreamPrivate* private_data =
      (struct ArrowBasicDeviceArrayStreamPrivate*)ArrowMalloc(
          sizeof(struct ArrowBasicDeviceArrayStreamPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  private_data->device = device;
  ArrowArrayStreamMove(array_stream, &private_data->naive_stream);

  device_array_stream->device_type = device->device_type;
  device_array_stream->get_schema = &ArrowDeviceBasicArrayStreamGetSchema;
  device_array_stream->get_next = &ArrowDeviceBasicArrayStreamGetNext;
  device_array_stream->get_last_error = &ArrowDeviceBasicArrayStreamGetLastError;
  device_array_stream->release = &ArrowDeviceBasicArrayStreamRelease;
  device_array_stream->private_data = private_data;
  return NANOARROW_OK;
}
