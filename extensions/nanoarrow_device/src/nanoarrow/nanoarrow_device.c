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
                                               struct ArrowBuffer* dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferInit(dst);
  dst->allocator = ArrowBufferAllocatorDefault();
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(dst, src.data.as_uint8, src.size_bytes));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferMove(struct ArrowDevice* device_src,
                                               struct ArrowBuffer* src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBuffer* dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  ArrowBufferMove(src, dst);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuBufferCopy(struct ArrowDevice* device_src,
                                               struct ArrowBufferView src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowBufferView dst) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  memcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
  return NANOARROW_OK;
}

static int ArrowDeviceCpuCopyRequired(struct ArrowDevice* device_src,
                                      struct ArrowArrayView* src,
                                      struct ArrowDevice* device_dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CPU) {
    return 0;
  } else {
    return -1;
  }
}

static ArrowErrorCode ArrowDeviceCpuSynchronize(struct ArrowDevice* device,
                                                void* sync_event,
                                                struct ArrowError* error) {
  switch (device->device_type) {
    case ARROW_DEVICE_CPU:
      if (sync_event != NULL) {
        ArrowErrorSet(error, "Expected NULL sync_event for ARROW_DEVICE_CPU but got %p",
                      sync_event);
        return EINVAL;
      } else {
        return NANOARROW_OK;
      }
    default:
      return device->synchronize_event(device, sync_event, error);
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
  device->copy_required = &ArrowDeviceCpuCopyRequired;
  device->synchronize_event = &ArrowDeviceCpuSynchronize;
  device->release = &ArrowDeviceCpuRelease;
  device->private_data = NULL;
}

#ifdef NANOARROW_DEVICE_WITH_METAL
struct ArrowDevice* ArrowDeviceMetalDefaultDevice(void);
#endif

#ifdef NANOARROW_DEVICE_WITH_CUDA
struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id);
#endif

struct ArrowDevice* ArrowDeviceResolve(ArrowDeviceType device_type, int64_t device_id) {
  if (device_type == ARROW_DEVICE_CPU && device_id == 0) {
    return ArrowDeviceCpu();
  }

#ifdef NANOARROW_DEVICE_WITH_METAL
  if (device_type == ARROW_DEVICE_METAL) {
    struct ArrowDevice* default_device = ArrowDeviceMetalDefaultDevice();
    if (device_id == default_device->device_id) {
      return default_device;
    }
  }
#endif

#ifdef NANOARROW_DEVICE_WITH_CUDA
  if (device_type == ARROW_DEVICE_CUDA || device_type == ARROW_DEVICE_CUDA_HOST) {
    return ArrowDeviceCuda(device_type, device_id);
  }
#endif

  return NULL;
}

ArrowErrorCode ArrowDeviceBufferInit(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst) {
  int result = device_dst->buffer_init(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_init(device_src, src, device_dst, dst);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferMove(struct ArrowDevice* device_src,
                                     struct ArrowBuffer* src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBuffer* dst) {
  int result = device_dst->buffer_move(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_move(device_src, src, device_dst, dst);
  }

  return result;
}

ArrowErrorCode ArrowDeviceBufferCopy(struct ArrowDevice* device_src,
                                     struct ArrowBufferView src,
                                     struct ArrowDevice* device_dst,
                                     struct ArrowBufferView dst) {
  int result = device_dst->buffer_copy(device_src, src, device_dst, dst);
  if (result == ENOTSUP) {
    result = device_src->buffer_copy(device_src, src, device_dst, dst);
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

void ArrowDeviceArrayViewInit(struct ArrowDeviceArrayView* device_array_view) {
  memset(device_array_view, 0, sizeof(struct ArrowDeviceArrayView));
}

void ArrowDeviceArrayViewReset(struct ArrowDeviceArrayView* device_array_view) {
  ArrowArrayViewReset(&device_array_view->array_view);
  device_array_view->device = NULL;
}

static ArrowErrorCode ArrowDeviceBufferGetInt32(struct ArrowDevice* device,
                                                struct ArrowBufferView buffer_view,
                                                int64_t i, int32_t* out) {
  struct ArrowBufferView out_view;
  out_view.data.as_int32 = out;
  out_view.size_bytes = sizeof(int32_t);

  struct ArrowBufferView device_buffer_view;
  device_buffer_view.data.as_int32 = buffer_view.data.as_int32 + i;
  device_buffer_view.size_bytes = sizeof(int32_t);
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceBufferCopy(device, device_buffer_view, ArrowDeviceCpu(), out_view));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceBufferGetInt64(struct ArrowDevice* device,
                                                struct ArrowBufferView buffer_view,
                                                int64_t i, int64_t* out) {
  struct ArrowBufferView out_view;
  out_view.data.as_int64 = out;
  out_view.size_bytes = sizeof(int64_t);

  struct ArrowBufferView device_buffer_view;
  device_buffer_view.data.as_int64 = buffer_view.data.as_int64 + i;
  device_buffer_view.size_bytes = sizeof(int64_t);
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceBufferCopy(device, device_buffer_view, ArrowDeviceCpu(), out_view));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceArrayViewValidateDefault(
    struct ArrowDevice* device, struct ArrowArrayView* array_view) {
  // Calculate buffer sizes or child lengths that require accessing the offsets
  // buffer. Unlike the nanoarrow core default validation, this just checks the
  // last buffer and doesn't set a nice error message (could implement those, too
  // later on).
  int64_t offset_plus_length = array_view->offset + array_view->length;
  int32_t last_offset32;
  int64_t last_offset64;

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt32(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset32));

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset32;
        } else if (array_view->buffer_views[2].size_bytes < last_offset32) {
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt64(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset64));

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset64;
        } else if (array_view->buffer_views[2].size_bytes < last_offset64) {
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      if (array_view->buffer_views[1].size_bytes != 0) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt32(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset32));
        if (array_view->children[0]->length < last_offset32) {
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LARGE_LIST:
      if (array_view->buffer_views[1].size_bytes != 0) {
        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferGetInt64(
            device, array_view->buffer_views[1], offset_plus_length, &last_offset64));
        if (array_view->children[0]->length < last_offset64) {
          return EINVAL;
        }
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceArrayViewValidateDefault(device, array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewSetArray(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error) {
  struct ArrowDevice* device =
      ArrowDeviceResolve(device_array->device_type, device_array->device_id);
  if (device == NULL) {
    ArrowErrorSet(error, "Can't resolve device with type %d and identifier %ld",
                  (int)device_array->device_type, (long)device_array->device_id);
    return EINVAL;
  }

  // Wait on device_array to synchronize with the CPU
  NANOARROW_RETURN_NOT_OK(
      device->synchronize_event(device, device_array->sync_event, error));

  // Set the device array device
  device_array_view->device = device;

  // nanoarrow's minimal validation is fine here (sets buffer sizes for non offset-buffer
  // types and errors for invalid ones)
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayMinimal(&device_array_view->array_view,
                                                        &device_array->array, error));
  // Run custom validator that copies memory to the CPU where required.
  // The custom implementation doesn't set nice error messages yet.
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowDeviceArrayViewValidateDefault(device, &device_array_view->array_view), error);

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceArrayViewCopyInternal(struct ArrowDevice* device_src,
                                                       struct ArrowArrayView* src,
                                                       struct ArrowDevice* device_dst,
                                                       struct ArrowArray* dst) {
  // Currently no attempt to minimize the amount of meory copied (i.e.,
  // by applying offset + length and copying potentially fewer bytes)
  dst->length = src->length;
  dst->offset = src->offset;
  dst->null_count = src->null_count;

  for (int i = 0; i < 3; i++) {
    if (src->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferInit(device_src, src->buffer_views[i],
                                                  device_dst, ArrowArrayBuffer(dst, i)));
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->children[i], device_dst, dst->children[i]));
  }

  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->dictionary, device_dst, dst->dictionary));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewCopy(struct ArrowDeviceArrayView* src,
                                        struct ArrowDevice* device_dst,
                                        struct ArrowDeviceArray* dst) {
  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(&tmp, &src->array_view, NULL));

  int result =
      ArrowDeviceArrayViewCopyInternal(src->device, &src->array_view, device_dst, &tmp);
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  result = ArrowArrayFinishBuilding(&tmp, NANOARROW_VALIDATION_LEVEL_MINIMAL, NULL);
  if (result != NANOARROW_OK) {
    tmp.release(&tmp);
    return result;
  }

  ArrowDeviceArrayInit(dst, device_dst);
  ArrowArrayMove(&tmp, &dst->array);
  dst->device_type = device_dst->device_type;
  dst->device_id = device_dst->device_id;
  return result;
}

int ArrowDeviceArrayViewCopyRequired(struct ArrowDeviceArrayView* src,
                                     struct ArrowDevice* device_dst) {
  // Can always move if device type and ID match
  if (src->device->device_type == device_dst->device_type &&
      src->device->device_id == device_dst->device_id) {
    return 0;
  }

  // Otherwise, see if the source device knows
  int result = src->device->copy_required(src->device, &src->array_view, device_dst);
  if (result == -1) {
    // Otherwise, see if the destination device knows
    result = device_dst->copy_required(src->device, &src->array_view, device_dst);
  }

  // If nobody knows, we have to copy
  return result != 0;
}

ArrowErrorCode ArrowDeviceArrayTryMove(struct ArrowDeviceArray* src,
                                       struct ArrowDeviceArrayView* src_view,
                                       struct ArrowDevice* device_dst,
                                       struct ArrowDeviceArray* dst) {
  if (ArrowDeviceArrayViewCopyRequired(src_view, device_dst)) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopy(src_view, device_dst, dst));
    src->array.release(&src->array);
    return NANOARROW_OK;
  } else {
    ArrowDeviceArrayMove(src, dst);
    dst->device_type = device_dst->device_type;
    dst->device_id = device_dst->device_id;
    return NANOARROW_OK;
  }
}
