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
#include <inttypes.h>

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_device.h"

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

static void ArrowDeviceArrayInitDefault(struct ArrowDevice* device,
                                        struct ArrowDeviceArray* device_array,
                                        struct ArrowArray* array) {
  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->device_type = device->device_type;
  device_array->device_id = device->device_id;
  ArrowArrayMove(array, &device_array->array);
}

static ArrowErrorCode ArrowDeviceCpuBufferInitAsync(struct ArrowDevice* device_src,
                                                    struct ArrowBufferView src,
                                                    struct ArrowDevice* device_dst,
                                                    struct ArrowBuffer* dst,
                                                    void* stream) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  if (stream != NULL) {
    return EINVAL;
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
                                               struct ArrowBufferView dst, void* stream) {
  if (device_dst->device_type != ARROW_DEVICE_CPU ||
      device_src->device_type != ARROW_DEVICE_CPU) {
    return ENOTSUP;
  }

  if (stream != NULL) {
    return EINVAL;
  }

  memcpy((uint8_t*)dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCpuSynchronize(struct ArrowDevice* device,
                                                void* sync_event, void* stream,
                                                struct ArrowError* error) {
  switch (device->device_type) {
    case ARROW_DEVICE_CPU:
      if (sync_event != NULL || stream != NULL) {
        ArrowErrorSet(error, "sync_event and stream must be NULL for ARROW_DEVICE_CPU");
        return EINVAL;
      } else {
        return NANOARROW_OK;
      }
    default:
      ArrowErrorSet(error, "Expected CPU device but got device type %d",
                    (int)device->device_id);
      return ENOTSUP;
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
  device->device_id = -1;
  device->array_init = NULL;
  device->array_move = NULL;
  device->buffer_init = &ArrowDeviceCpuBufferInitAsync;
  device->buffer_move = &ArrowDeviceCpuBufferMove;
  device->buffer_copy = &ArrowDeviceCpuBufferCopy;
  device->synchronize_event = &ArrowDeviceCpuSynchronize;
  device->release = &ArrowDeviceCpuRelease;
  device->private_data = NULL;
}

struct ArrowDevice* ArrowDeviceResolve(ArrowDeviceType device_type, int64_t device_id) {
  NANOARROW_UNUSED(device_id);
  if (device_type == ARROW_DEVICE_CPU) {
    return ArrowDeviceCpu();
  }

  if (device_type == ARROW_DEVICE_METAL) {
    struct ArrowDevice* default_device = ArrowDeviceMetalDefaultDevice();
    if (device_id == default_device->device_id) {
      return default_device;
    }
  }

  if (device_type == ARROW_DEVICE_CUDA || device_type == ARROW_DEVICE_CUDA_HOST) {
    return ArrowDeviceCuda(device_type, device_id);
  }

  return NULL;
}

ArrowErrorCode ArrowDeviceArrayInitAsync(struct ArrowDevice* device,
                                         struct ArrowDeviceArray* device_array,
                                         struct ArrowArray* array, void* sync_event,
                                         void* stream) {
  if (device->array_init != NULL) {
    return device->array_init(device, device_array, array, sync_event, stream);
  }

  // Sync event and stream aren't handled by the fallback implementation
  if (sync_event != NULL || stream != NULL) {
    return EINVAL;
  }

  ArrowDeviceArrayInitDefault(device, device_array, array);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceBufferInitAsync(struct ArrowDevice* device_src,
                                          struct ArrowBufferView src,
                                          struct ArrowDevice* device_dst,
                                          struct ArrowBuffer* dst, void* stream) {
  int result = device_dst->buffer_init(device_src, src, device_dst, dst, stream);
  if (result == ENOTSUP) {
    result = device_src->buffer_init(device_src, src, device_dst, dst, stream);
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

ArrowErrorCode ArrowDeviceBufferCopyAsync(struct ArrowDevice* device_src,
                                          struct ArrowBufferView src,
                                          struct ArrowDevice* device_dst,
                                          struct ArrowBufferView dst, void* stream) {
  int result = device_dst->buffer_copy(device_src, src, device_dst, dst, stream);
  if (result == ENOTSUP) {
    result = device_src->buffer_copy(device_src, src, device_dst, dst, stream);
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
  int result = ArrowDeviceArrayInit(private_data->device, device_array, &tmp, NULL);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

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
  ArrowArrayStreamRelease(&private_data->naive_stream);
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

ArrowErrorCode ArrowDeviceArrayViewSetArrayMinimal(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    struct ArrowError* error) {
  // Resolve device
  struct ArrowDevice* device =
      ArrowDeviceResolve(device_array->device_type, device_array->device_id);
  if (device == NULL) {
    ArrowErrorSet(error,
                  "Can't resolve device with type %" PRId32 " and identifier %" PRId64,
                  device_array->device_type, device_array->device_id);
    return EINVAL;
  }

  // Set the device array device
  device_array_view->device = device;

  // Populate the array_view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayMinimal(&device_array_view->array_view,
                                                        &device_array->array, error));

  // Populate the sync_event
  device_array_view->sync_event = device_array->sync_event;

  return NANOARROW_OK;
}

// Walks the tree of arrays to count the number of buffers with unknown size
// and the number of bytes we need to copy from a device buffer to find it.
static ArrowErrorCode ArrowDeviceArrayViewWalkUnknownBufferSizes(
    struct ArrowArrayView* array_view, int64_t* offset_buffer_size) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->length == 0 || array_view->buffer_views[1].size_bytes == 0) {
        array_view->buffer_views[2].size_bytes = 0;
      } else if (array_view->buffer_views[2].size_bytes == -1) {
        *offset_buffer_size += array_view->layout.element_size_bits[1] / 8;
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewWalkUnknownBufferSizes(
        array_view->children[i], offset_buffer_size));
  }

  // ...and for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewWalkUnknownBufferSizes(
        array_view->dictionary, offset_buffer_size));
  }

  return NANOARROW_OK;
}

// Walks the tree of arrays and launches an async copy of the relevant
// item in the array's offset buffer to the temporary buffer we've just
// allocated to collect these values.
static ArrowErrorCode ArrowDeviceArrayViewResolveUnknownBufferSizesAsync(
    struct ArrowDevice* device, struct ArrowArrayView* array_view,
    uint8_t** offset_value_dst, void* stream) {
  int64_t offset_plus_length = array_view->offset + array_view->length;

  struct ArrowBufferView src_view;
  struct ArrowBufferView dst_view;

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[2].size_bytes == -1) {
        src_view.data.as_int32 =
            array_view->buffer_views[1].data.as_int32 + offset_plus_length;
        src_view.size_bytes = sizeof(int32_t);
        dst_view.data.as_uint8 = *offset_value_dst;
        dst_view.size_bytes = sizeof(int32_t);

        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferCopyAsync(
            device, src_view, ArrowDeviceCpu(), dst_view, stream));

        (*offset_value_dst) += sizeof(int32_t);
      }
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[2].size_bytes == -1) {
        src_view.data.as_int64 =
            array_view->buffer_views[1].data.as_int64 + offset_plus_length;
        src_view.size_bytes = sizeof(int64_t);
        dst_view.data.as_uint8 = *offset_value_dst;
        dst_view.size_bytes = sizeof(int64_t);

        NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferCopyAsync(
            device, src_view, ArrowDeviceCpu(), dst_view, stream));

        (*offset_value_dst) += sizeof(int64_t);
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewResolveUnknownBufferSizesAsync(
        device, array_view->children[i], offset_value_dst, stream));
  }

  // ...and for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewResolveUnknownBufferSizesAsync(
        device, array_view->dictionary, offset_value_dst, stream));
  }

  return NANOARROW_OK;
}

// After synchronizing the stream with the CPU to ensure that all of the
// buffer sizes have been copied to the our temporary buffer, relay them
// back to the appropriate buffer view so that the buffer copier can
// do its thing.
static void ArrowDeviceArrayViewCollectUnknownBufferSizes(
    struct ArrowArrayView* array_view, uint8_t** offset_value_dst) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[2].size_bytes == -1) {
        int32_t size_bytes_32;
        memcpy(&size_bytes_32, *offset_value_dst, sizeof(int32_t));
        array_view->buffer_views[2].size_bytes = size_bytes_32;
        (*offset_value_dst) += sizeof(int32_t);
      }
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[2].size_bytes == -1) {
        memcpy(&array_view->buffer_views[2].size_bytes, *offset_value_dst,
               sizeof(int64_t));
        (*offset_value_dst) += sizeof(int64_t);
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    ArrowDeviceArrayViewCollectUnknownBufferSizes(array_view->children[i],
                                                  offset_value_dst);
  }

  // ...and for dictionary
  if (array_view->dictionary != NULL) {
    ArrowDeviceArrayViewCollectUnknownBufferSizes(array_view->dictionary,
                                                  offset_value_dst);
  }
}

static ArrowErrorCode ArrowDeviceArrayViewEnsureBufferSizesAsync(
    struct ArrowDeviceArrayView* device_array_view, void* stream,
    struct ArrowError* error) {
  // Walk the tree of arrays to check for buffers whose size we don't know
  int64_t temp_buffer_length_bytes_required = 0;
  NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewWalkUnknownBufferSizes(
      &device_array_view->array_view, &temp_buffer_length_bytes_required));

  // If there are no such arrays (e.g., there are no string or binary arrays in the tree),
  // we don't have to do anything extra
  if (temp_buffer_length_bytes_required == 0) {
    return NANOARROW_OK;
  }

  // Ensure that the stream provided waits on the array's sync event
  NANOARROW_RETURN_NOT_OK(device_array_view->device->synchronize_event(
      device_array_view->device, device_array_view->sync_event, stream, error));

  // Allocate a buffer big enough to hold all the offset values we need to
  // copy from the GPU
  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferResize(&buffer, temp_buffer_length_bytes_required, 0));

  uint8_t* cursor = buffer.data;
  int result = ArrowDeviceArrayViewResolveUnknownBufferSizesAsync(
      device_array_view->device, &device_array_view->array_view, &cursor, stream);
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&buffer);
    return result;
  }

  NANOARROW_DCHECK(cursor == (buffer.data + buffer.size_bytes));

  // Synchronize the stream with the CPU
  result = device_array_view->device->synchronize_event(device_array_view->device, NULL,
                                                        stream, error);

  // Collect the values from the temporary buffer
  cursor = buffer.data;
  ArrowDeviceArrayViewCollectUnknownBufferSizes(&device_array_view->array_view, &cursor);
  NANOARROW_DCHECK(cursor == (buffer.data + buffer.size_bytes));
  ArrowBufferReset(&buffer);

  return result;
}

ArrowErrorCode ArrowDeviceArrayViewSetArrayAsync(
    struct ArrowDeviceArrayView* device_array_view, struct ArrowDeviceArray* device_array,
    void* stream, struct ArrowError* error) {
  // Populate the array view with all information accessible from the CPU
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceArrayViewSetArrayMinimal(device_array_view, device_array, error));

  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceArrayViewEnsureBufferSizesAsync(device_array_view, stream, error));

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceArrayViewCopyInternal(struct ArrowDevice* device_src,
                                                       struct ArrowArrayView* src,
                                                       struct ArrowDevice* device_dst,
                                                       struct ArrowArray* dst,
                                                       void* stream) {
  // Currently no attempt to minimize the amount of memory copied (i.e.,
  // by applying offset + length and copying potentially fewer bytes)
  dst->length = src->length;
  dst->offset = src->offset;
  dst->null_count = src->null_count;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (src->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(ArrowDeviceBufferInitAsync(
        device_src, src->buffer_views[i], device_dst, ArrowArrayBuffer(dst, i), stream));
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->children[i], device_dst, dst->children[i], stream));
  }

  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewCopyInternal(
        device_src, src->dictionary, device_dst, dst->dictionary, stream));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowDeviceArrayViewCopyAsync(struct ArrowDeviceArrayView* src,
                                             struct ArrowDevice* device_dst,
                                             struct ArrowDeviceArray* dst, void* stream) {
  // Ensure src has all buffer sizes defined
  NANOARROW_RETURN_NOT_OK(ArrowDeviceArrayViewEnsureBufferSizesAsync(src, stream, NULL));

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(&tmp, &src->array_view, NULL));

  int result = ArrowDeviceArrayViewCopyInternal(src->device, &src->array_view, device_dst,
                                                &tmp, stream);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  // If we are copying to the CPU, we need to synchronize the stream because we
  // can't populate a sync event for a CPU array.
  if (device_dst->device_type == ARROW_DEVICE_CPU) {
    result = src->device->synchronize_event(src->device, NULL, stream, NULL);
    if (result != NANOARROW_OK) {
      ArrowArrayRelease(&tmp);
      return result;
    }

    stream = NULL;
  }

  result = ArrowArrayFinishBuilding(&tmp, NANOARROW_VALIDATION_LEVEL_MINIMAL, NULL);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  result = ArrowDeviceArrayInitAsync(device_dst, dst, &tmp, NULL, stream);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  return result;
}

ArrowErrorCode ArrowDeviceArrayMoveToDevice(struct ArrowDeviceArray* src,
                                            struct ArrowDevice* device_dst,
                                            struct ArrowDeviceArray* dst) {
  // Can always move from the same device to the same device
  if (src->device_type == device_dst->device_type &&
      src->device_id == device_dst->device_id) {
    ArrowDeviceArrayMove(src, dst);
    return NANOARROW_OK;
  }

  struct ArrowDevice* device_src = ArrowDeviceResolve(src->device_type, src->device_id);
  if (device_src == NULL) {
    return EINVAL;
  }

  // See if the source knows how to move
  int result;
  if (device_src->array_move != NULL) {
    result = device_src->array_move(device_src, src, device_dst, dst);
    if (result != ENOTSUP) {
      return result;
    }
  }

  // See if the destination knows how to move
  if (device_dst->array_move != NULL) {
    NANOARROW_RETURN_NOT_OK(device_dst->array_move(device_src, src, device_dst, dst));
  }

  return NANOARROW_OK;
}

#if !defined(NANOARROW_DEVICE_WITH_CUDA)
struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id) {
  NANOARROW_UNUSED(device_type);
  NANOARROW_UNUSED(device_id);

  return NULL;
}
#endif

#if !defined(NANOARROW_DEVICE_WITH_METAL)
struct ArrowDevice* ArrowDeviceMetalDefaultDevice(void) { return NULL; }

ArrowErrorCode ArrowDeviceMetalInitDefaultDevice(struct ArrowDevice* device,
                                                 struct ArrowError* error) {
  NANOARROW_UNUSED(device);

  ArrowErrorSet(error, "nanoarrow_device not built with Metal support");
  return ENOTSUP;
}

ArrowErrorCode ArrowDeviceMetalInitBuffer(struct ArrowBuffer* buffer) {
  NANOARROW_UNUSED(buffer);
  return ENOTSUP;
}

ArrowErrorCode ArrowDeviceMetalAlignArrayBuffers(struct ArrowArray* array) {
  NANOARROW_UNUSED(array);
  return ENOTSUP;
}
#endif
