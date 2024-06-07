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

#include <inttypes.h>

#include <cuda.h>

#include "nanoarrow/nanoarrow_device.h"

#define NANOARROW_CUDA_DEFAULT_STREAM ((CUstream)0)

static inline void ArrowDeviceCudaSetError(CUresult err, const char* op,
                                           struct ArrowError* error) {
  if (error == NULL) {
    return;
  }

  const char* name = NULL;
  CUresult err_result = cuGetErrorName(err, &name);
  if (err_result != CUDA_SUCCESS || name == NULL) {
    name = "name unknown";
  }

  const char* description = NULL;
  err_result = cuGetErrorString(err, &description);
  if (err_result != CUDA_SUCCESS || description == NULL) {
    description = "description unknown";
  }

  ArrowErrorSet(error, "[%s][%s] %s", op, name, description);
}

#define _NANOARROW_CUDA_RETURN_NOT_OK_IMPL(NAME, EXPR, OP, ERROR) \
  do {                                                            \
    CUresult NAME = (EXPR);                                       \
    if (NAME != CUDA_SUCCESS) {                                   \
      ArrowDeviceCudaSetError(NAME, OP, ERROR);                   \
      return EIO;                                                 \
    }                                                             \
  } while (0)

#define NANOARROW_CUDA_RETURN_NOT_OK(EXPR, OP, ERROR)                                    \
  _NANOARROW_CUDA_RETURN_NOT_OK_IMPL(_NANOARROW_MAKE_NAME(cuda_err_, __COUNTER__), EXPR, \
                                     OP, ERROR)

#if defined(NANOARROW_DEBUG)
#define _NANOARROW_CUDA_ASSERT_OK_IMPL(NAME, EXPR, EXPR_STR)           \
  do {                                                                 \
    const CUresult NAME = (EXPR);                                      \
    if (NAME != CUDA_SUCCESS) NANOARROW_PRINT_AND_DIE(NAME, EXPR_STR); \
  } while (0)
#define NANOARROW_CUDA_ASSERT_OK(EXPR)                                                   \
  _NANOARROW_CUDA_ASSERT_OK_IMPL(_NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR, \
                                 #EXPR)
#else
#define NANOARROW_CUDA_ASSERT_OK(EXPR) (void)(EXPR)
#endif

struct ArrowDeviceCudaPrivate {
  CUdevice cu_device;
  CUcontext cu_context;
};

struct ArrowDeviceCudaAllocatorPrivate {
  ArrowDeviceType device_type;
  int64_t device_id;
  // When moving a buffer from CUDA_HOST to CUDA, the pointer used to access
  // the data changes but the pointer needed to pass to cudaFreeHost does not
  void* allocated_ptr;
};

static void ArrowDeviceCudaDeallocator(struct ArrowBufferAllocator* allocator,
                                       uint8_t* ptr, int64_t old_size) {
  struct ArrowDeviceCudaAllocatorPrivate* allocator_private =
      (struct ArrowDeviceCudaAllocatorPrivate*)allocator->private_data;

  switch (allocator_private->device_type) {
    case ARROW_DEVICE_CUDA:
      cuMemFree((CUdeviceptr)allocator_private->allocated_ptr);
      break;
    case ARROW_DEVICE_CUDA_HOST:
      NANOARROW_CUDA_ASSERT_OK(cuMemFreeHost(allocator_private->allocated_ptr));
      break;
    default:
      break;
  }

  ArrowFree(allocator_private);
}

static ArrowErrorCode ArrowDeviceCudaAllocateBufferAsync(struct ArrowDevice* device,
                                                         struct ArrowBuffer* buffer,
                                                         int64_t size_bytes,
                                                         CUstream hstream) {
  struct ArrowDeviceCudaPrivate* private_data =
      (struct ArrowDeviceCudaPrivate*)device->private_data;

  NANOARROW_CUDA_RETURN_NOT_OK(cuCtxPushCurrent(private_data->cu_context),
                               "cuCtxPushCurrent", NULL);
  CUcontext unused;  // needed for cuCtxPopCurrent()

  struct ArrowDeviceCudaAllocatorPrivate* allocator_private =
      (struct ArrowDeviceCudaAllocatorPrivate*)ArrowMalloc(
          sizeof(struct ArrowDeviceCudaAllocatorPrivate));
  if (allocator_private == NULL) {
    NANOARROW_CUDA_ASSERT_OK(cuCtxPopCurrent(&unused));
    return ENOMEM;
  }

  CUresult err;
  void* ptr = NULL;
  const char* op = "";
  switch (device->device_type) {
    case ARROW_DEVICE_CUDA: {
      CUdeviceptr dptr = 0;

      // cuMemalloc requires non-zero size_bytes
      if (size_bytes > 0) {
        err = cuMemAllocAsync(&dptr, (size_t)size_bytes, hstream);
      } else {
        err = CUDA_SUCCESS;
      }

      ptr = (void*)dptr;
      op = "cuMemAlloc";
      break;
    }
    case ARROW_DEVICE_CUDA_HOST:
      err = cuMemAllocHost(&ptr, (size_t)size_bytes);
      op = "cuMemAllocHost";
      break;
    default:
      cuCtxPopCurrent(&unused);
      ArrowFree(allocator_private);
      return EINVAL;
  }

  if (err != CUDA_SUCCESS) {
    NANOARROW_CUDA_ASSERT_OK(cuCtxPopCurrent(&unused));
    ArrowFree(allocator_private);
    return EIO;
  }

  allocator_private->device_id = device->device_id;
  allocator_private->device_type = device->device_type;
  allocator_private->allocated_ptr = ptr;

  buffer->data = (uint8_t*)ptr;
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = size_bytes;
  buffer->allocator =
      ArrowBufferDeallocator(&ArrowDeviceCudaDeallocator, allocator_private);

  NANOARROW_CUDA_ASSERT_OK(cuCtxPopCurrent(&unused));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaAllocateBuffer(struct ArrowDevice* device,
                                                    struct ArrowBuffer* buffer,
                                                    int64_t size_bytes) {
  return ArrowDeviceCudaAllocateBufferAsync(device, buffer, size_bytes,
                                            NANOARROW_CUDA_DEFAULT_STREAM);
}

struct ArrowDeviceCudaArrayPrivate {
  struct ArrowArray parent;
  CUevent cu_event;
};

static void ArrowDeviceCudaArrayRelease(struct ArrowArray* array) {
  struct ArrowDeviceCudaArrayPrivate* private_data =
      (struct ArrowDeviceCudaArrayPrivate*)array->private_data;

  if (private_data->cu_event != NULL) {
    NANOARROW_CUDA_ASSERT_OK(cuEventDestroy(private_data->cu_event));
  }

  ArrowArrayRelease(&private_data->parent);
  ArrowFree(private_data);
  array->release = NULL;
}

static ArrowErrorCode ArrowDeviceCudaArrayInit(struct ArrowDevice* device,
                                               struct ArrowDeviceArray* device_array,
                                               struct ArrowArray* array,
                                               void* sync_event) {
  struct ArrowDeviceCudaPrivate* device_private =
      (struct ArrowDeviceCudaPrivate*)device->private_data;
  // One can create an event with cuEventCreate(&cu_event, CU_EVENT_DEFAULT);
  // Requires cuCtxPushCurrent() + cuEventCreate() + cuCtxPopCurrent()

  struct ArrowDeviceCudaArrayPrivate* private_data =
      (struct ArrowDeviceCudaArrayPrivate*)ArrowMalloc(
          sizeof(struct ArrowDeviceCudaArrayPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->array = *array;
  device_array->array.private_data = private_data;
  device_array->array.release = &ArrowDeviceCudaArrayRelease;
  ArrowArrayMove(array, &private_data->parent);

  device_array->device_id = device->device_id;
  device_array->device_type = device->device_type;

  if (sync_event != NULL) {
    private_data->cu_event = *((CUevent*)sync_event);
    device_array->sync_event = sync_event;
  } else {
    private_data->cu_event = NULL;
    device_array->sync_event = NULL;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaBufferCopyInternal(
    struct ArrowDevice* device_src, struct ArrowBufferView src,
    struct ArrowDevice* device_dst, struct ArrowBufferView dst, int* n_pop_context,
    struct ArrowError* error, CUstream hstream) {
  // Note: the device_src/sync event must be synchronized before calling these methods,
  // even though the cuMemcpyXXX() functions may do this automatically in some cases.

  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    struct ArrowDeviceCudaPrivate* dst_private =
        (struct ArrowDeviceCudaPrivate*)device_dst->private_data;
    NANOARROW_CUDA_RETURN_NOT_OK(cuCtxPushCurrent(dst_private->cu_context),
                                 "cuCtxPushCurrent", error);
    (*n_pop_context)++;

    NANOARROW_CUDA_RETURN_NOT_OK(
        cuMemcpyHtoDAsync((CUdeviceptr)dst.data.data, src.data.data,
                          (size_t)src.size_bytes, hstream),
        "cuMemcpyHtoD", error);

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA &&
             device_src->device_id == device_dst->device_id) {
    struct ArrowDeviceCudaPrivate* dst_private =
        (struct ArrowDeviceCudaPrivate*)device_dst->private_data;

    NANOARROW_CUDA_RETURN_NOT_OK(cuCtxPushCurrent(dst_private->cu_context),
                                 "cuCtxPushCurrent", error);
    (*n_pop_context)++;

    NANOARROW_CUDA_RETURN_NOT_OK(
        cuMemcpyDtoDAsync((CUdeviceptr)dst.data.data, (CUdeviceptr)src.data.data,
                          (size_t)src.size_bytes, hstream),
        "cuMemcpytoD", error);

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA) {
    struct ArrowDeviceCudaPrivate* src_private =
        (struct ArrowDeviceCudaPrivate*)device_src->private_data;
    struct ArrowDeviceCudaPrivate* dst_private =
        (struct ArrowDeviceCudaPrivate*)device_dst->private_data;

    NANOARROW_CUDA_RETURN_NOT_OK(
        cuMemcpyPeerAsync((CUdeviceptr)dst.data.data, dst_private->cu_context,
                          (CUdeviceptr)src.data.data, src_private->cu_context,
                          (size_t)src.size_bytes, hstream),
        "cuMemcpyPeer", error);

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    struct ArrowDeviceCudaPrivate* src_private =
        (struct ArrowDeviceCudaPrivate*)device_src->private_data;

    NANOARROW_CUDA_RETURN_NOT_OK(cuCtxPushCurrent(src_private->cu_context),
                                 "cuCtxPushCurrent", error);
    (*n_pop_context)++;
    NANOARROW_CUDA_RETURN_NOT_OK(
        cuMemcpyDtoHAsync((void*)dst.data.data, (CUdeviceptr)src.data.data,
                          (size_t)src.size_bytes, hstream),
        "cuMemcpyDtoH", error);

  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy((void*)dst.data.data, src.data.data, (size_t)src.size_bytes);

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy((void*)dst.data.data, src.data.data, (size_t)src.size_bytes);

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy((void*)dst.data.data, src.data.data, (size_t)src.size_bytes);

  } else {
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaBufferCopyAsync(struct ArrowDevice* device_src,
                                                     struct ArrowBufferView src,
                                                     struct ArrowDevice* device_dst,
                                                     struct ArrowBufferView dst,
                                                     CUstream hstream) {
  int n_pop_context = 0;
  struct ArrowError error;

  int result = ArrowDeviceCudaBufferCopyInternal(device_src, src, device_dst, dst,
                                                 &n_pop_context, &error, hstream);
  for (int i = 0; i < n_pop_context; i++) {
    CUcontext unused;
    NANOARROW_CUDA_ASSERT_OK(cuCtxPopCurrent(&unused));
  }

  return result;
}

static ArrowErrorCode ArrowDeviceCudaBufferCopy(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBufferView dst) {
  return ArrowDeviceCudaBufferCopyAsync(device_src, src, device_dst, dst,
                                        NANOARROW_CUDA_DEFAULT_STREAM);
}

static ArrowErrorCode ArrowDeviceCudaBufferInitAsync(struct ArrowDevice* device_src,
                                                     struct ArrowBufferView src,
                                                     struct ArrowDevice* device_dst,
                                                     struct ArrowBuffer* dst,
                                                     CUstream hstream) {
  struct ArrowBuffer tmp;

  switch (device_dst->device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
      NANOARROW_RETURN_NOT_OK(
          ArrowDeviceCudaAllocateBufferAsync(device_dst, &tmp, src.size_bytes, hstream));
      break;
    case ARROW_DEVICE_CPU:
      ArrowBufferInit(&tmp);
      NANOARROW_RETURN_NOT_OK(ArrowBufferResize(&tmp, src.size_bytes, 0));
      break;
    default:
      return ENOTSUP;
  }

  struct ArrowBufferView tmp_view;
  tmp_view.data.data = tmp.data;
  tmp_view.size_bytes = tmp.size_bytes;
  int result =
      ArrowDeviceCudaBufferCopyAsync(device_src, src, device_dst, tmp_view, hstream);
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&tmp);
    return result;
  }

  ArrowBufferMove(&tmp, dst);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaBufferInit(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBuffer* dst) {
  return ArrowDeviceCudaBufferInitAsync(device_src, src, device_dst, dst,
                                        NANOARROW_CUDA_DEFAULT_STREAM);
}

static ArrowErrorCode ArrowDeviceCudaSynchronize(struct ArrowDevice* device,
                                                 void* sync_event,
                                                 struct ArrowError* error) {
  if (sync_event == NULL) {
    return NANOARROW_OK;
  }

  if (device->device_type != ARROW_DEVICE_CUDA &&
      device->device_type != ARROW_DEVICE_CUDA_HOST) {
    return ENOTSUP;
  }

  // Memory for cuda_event is owned by the ArrowArray member of the ArrowDeviceArray
  CUevent* cuda_event = (CUevent*)sync_event;
  NANOARROW_CUDA_RETURN_NOT_OK(cuEventSynchronize(*cuda_event), "cuEventSynchronize",
                               error);

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaArrayMove(struct ArrowDevice* device_src,
                                               struct ArrowDeviceArray* src,
                                               struct ArrowDevice* device_dst,
                                               struct ArrowDeviceArray* dst) {
  // Note that the case where the devices are the same is handled before this

  if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
      device_dst->device_type == ARROW_DEVICE_CPU) {
    // Move: the array's release callback is responsible for cudaFreeHost or
    // deregistration (or perhaps this has been handled at a higher level).
    // We do have to wait on the sync event, though, because this has to be NULL
    // for a CPU device array.
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaSynchronize(device_src, src->sync_event, NULL));
    ArrowDeviceArrayMove(src, dst);
    dst->device_type = device_dst->device_type;
    dst->device_id = device_dst->device_id;
    dst->sync_event = NULL;

    return NANOARROW_OK;
  }

  // TODO: We can theoretically also do a move from CUDA_HOST to CUDA

  return ENOTSUP;
}

static ArrowErrorCode ArrowDeviceCudaResolveBufferSizesAsync(
    struct ArrowDevice* device, struct ArrowArrayView* array_view, CUstream hstream) {
  // Calculate buffer sizes that require accessing the offset buffer
  // (at this point all other sizes have been resolved).
  int64_t offset_plus_length = array_view->offset + array_view->length;

  struct ArrowBufferView src_view;
  struct ArrowBufferView dst_view;

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      array_view->buffer_views[2].size_bytes = 0;

      // Note that this strategy (copying four bytes into an int64_t)
      // only works on little-endian (should be checked at a higher level).
      if (array_view->buffer_views[2].size_bytes == -1) {
        src_view.data.as_int32 =
            array_view->buffer_views[1].data.as_int32 + offset_plus_length;
        src_view.size_bytes = sizeof(int32_t);

        dst_view.data.data = &(array_view->buffer_views[2].size_bytes);
        dst_view.size_bytes = sizeof(int32_t);

        NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaBufferCopyAsync(
            device, src_view, ArrowDeviceCpu(), dst_view, hstream));
      }

      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      array_view->buffer_views[2].size_bytes = 0;

      if (array_view->buffer_views[2].size_bytes == -1) {
        src_view.data.as_int64 =
            array_view->buffer_views[1].data.as_int64 + offset_plus_length;
        src_view.size_bytes = sizeof(int64_t);

        dst_view.data.data = &(array_view->buffer_views[2].size_bytes);
        dst_view.size_bytes = sizeof(int64_t);

        NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaBufferCopyAsync(
            device, src_view, ArrowDeviceCpu(), dst_view, hstream));
      }
      break;
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaResolveBufferSizesAsync(device, array_view->children[i], hstream));
  }

  // ...and for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaResolveBufferSizesAsync(device, array_view->dictionary, hstream));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaArrayViewCopyBuffers(
    struct ArrowDevice* device_src, struct ArrowArrayView* src,
    struct ArrowDevice* device_dst, struct ArrowArray* dst, CUstream hstream,
    int64_t* unknown_buffer_size_count) {
  // Currently no attempt to minimize the amount of memory copied (i.e.,
  // by applying offset + length and copying potentially fewer bytes)
  dst->length = src->length;
  dst->offset = src->offset;
  dst->null_count = src->null_count;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (src->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    struct ArrowBufferView src_view = src->buffer_views[i];
    int64_t buffer_size = src_view.size_bytes;

    // No need to initialize a buffer of zero bytes
    if (buffer_size == 0) {
      continue;
    }

    // buffer_size will be -1 if it is unknown (i.e., string/binary
    // data buffer when copying from CUDA -> CPU)
    if (unknown_buffer_size_count != NULL && buffer_size < 0) {
      (*unknown_buffer_size_count)++;
      continue;
    } else if (buffer_size < 0) {
      return EINVAL;
    }

    struct ArrowBuffer* buffer_dst = ArrowArrayBuffer(dst, i);

    // Don't initialize a buffer that has already been initialized
    if (buffer_dst->size_bytes > 0) {
      continue;
    }

    // Otherwise, initialize the buffer
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaBufferInitAsync(
        device_src, src_view, device_dst, buffer_dst, hstream));
  }

  // Recurse for children
  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaArrayViewCopyBuffers(
        device_src, src->children[i], device_dst, dst->children[i], hstream,
        unknown_buffer_size_count));
  }

  // ...and for dictionary
  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaArrayViewCopyBuffers(
        device_src, src->dictionary, device_dst, dst->dictionary, hstream,
        unknown_buffer_size_count));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaCreateStream(struct ArrowDevice* src,
                                                  struct ArrowDevice* dst, CUstream* out,
                                                  struct ArrowError* error) {
  struct ArrowDeviceCudaPrivate* private_data;

  if (dst->device_type == ARROW_DEVICE_CUDA) {
    private_data = (struct ArrowDeviceCudaPrivate*)dst->private_data;
  } else if (src->device_type == ARROW_DEVICE_CUDA) {
    private_data = (struct ArrowDeviceCudaPrivate*)src->private_data;
  } else {
    return NANOARROW_OK;
  }

  NANOARROW_CUDA_RETURN_NOT_OK(cuCtxPushCurrent(private_data->cu_context),
                               "cuCtxPushCurrent", NULL);
  CUresult err = cuStreamCreate(out, CU_STREAM_NON_BLOCKING);
  CUcontext unused;
  NANOARROW_CUDA_ASSERT_OK(cuCtxPopCurrent(&unused));
  NANOARROW_CUDA_RETURN_NOT_OK(err, "cuStreamCreate", error);

  return NANOARROW_OK;
}

static void ArrowDeviceCudaDestroyStream(CUstream hstream) {
  if (hstream != NANOARROW_CUDA_DEFAULT_STREAM) {
    NANOARROW_CUDA_ASSERT_OK(cuStreamDestroy(hstream));
  }
}

static ArrowErrorCode ArrowDeviceCudaArrayViewCopyInternal(
    struct ArrowDevice* device_src, struct ArrowDeviceArrayView* src,
    struct ArrowDevice* device_dst, struct ArrowArray* dst, struct ArrowError* error) {
  CUresult err;
  int code;
  int64_t unknown_buffer_size_count = 0;

  // Create a stream to do the first (and maybe only) round of buffer copying.
  // The ArrowDeviceArrayView should probably have a sync_event so that we can
  // force the stream we just created to wait on the event.
  CUstream hstream = NANOARROW_CUDA_DEFAULT_STREAM;
  NANOARROW_RETURN_NOT_OK(
      ArrowDeviceCudaCreateStream(device_src, device_dst, &hstream, error));

  code = ArrowDeviceCudaArrayViewCopyBuffers(device_src, &src->array_view, device_dst,
                                             dst, hstream, &unknown_buffer_size_count);
  if (code != NANOARROW_OK) {
    ArrowDeviceCudaDestroyStream(hstream);
    ArrowErrorSet(error, "ArrowDeviceCudaArrayViewCopyBuffers() failed");
    return code;
  }

  CUstream hstream_data = NANOARROW_CUDA_DEFAULT_STREAM;

  if (unknown_buffer_size_count > 0) {
    // If buffers with unknown size were encountered, we need to copy all of their
    // sizes to the CPU, synchronize, then copy all of the buffers once the CPU
    // knows how many bytes from each buffer need to be copied.
    code = ArrowDeviceCudaCreateStream(device_src, device_dst, &hstream_data, error);
    if (code != NANOARROW_OK) {
      ArrowDeviceCudaDestroyStream(hstream);
      NANOARROW_CUDA_RETURN_NOT_OK(err, "cudaStreamCreate", error);
    }

    // Issue all required async copies
    code = ArrowDeviceCudaResolveBufferSizesAsync(device_src, &src->array_view,
                                                  hstream_data);
    if (code != NANOARROW_OK) {
      ArrowDeviceCudaDestroyStream(hstream);
      ArrowErrorSet(error, "ArrowDeviceCudaResolveBufferSizesAsync() failed");
      return code;
    }

    // Wait for them all to complete
    err = cuStreamSynchronize(hstream);
    if (err != CUDA_SUCCESS) {
      ArrowDeviceCudaDestroyStream(hstream);
      ArrowDeviceCudaDestroyStream(hstream_data);
      NANOARROW_CUDA_RETURN_NOT_OK(err, "cuStreamSynchronize()", error);
    }

    // Issue async copies for data buffers. This will skip buffers that are already
    // being copied.
    unknown_buffer_size_count = 0;
    code = ArrowDeviceCudaArrayViewCopyBuffers(device_src, &src->array_view, device_dst,
                                               dst, hstream, &unknown_buffer_size_count);
    if (code != NANOARROW_OK) {
      ArrowDeviceCudaDestroyStream(hstream);
      ArrowDeviceCudaDestroyStream(hstream_data);
      ArrowErrorSet(error,
                    "ArrowDeviceCudaArrayViewCopyInternal() for data buffers failed");
      return code;
    }

    // There should not have been any buffers with unknown size in this round of copying
    NANOARROW_DCHECK(unknown_buffer_size_count == 0);
  }

  // TODO: Populate dst->sync_event and record the copying work
  // instead of synchronizing here.
  if (hstream != NANOARROW_CUDA_DEFAULT_STREAM) {
    err = cuStreamSynchronize(hstream);
    ArrowDeviceCudaDestroyStream(hstream);
    if (err != CUDA_SUCCESS) {
      ArrowDeviceCudaDestroyStream(hstream_data);
      NANOARROW_CUDA_RETURN_NOT_OK(err, "cuStreamSynchronize(hstream)", error);
    }
  }

  if (hstream_data != NANOARROW_CUDA_DEFAULT_STREAM) {
    err = cuStreamSynchronize(hstream_data);
    ArrowDeviceCudaDestroyStream(hstream_data);
    if (err != CUDA_SUCCESS) {
      NANOARROW_CUDA_RETURN_NOT_OK(err, "cuStreamSynchronize(hstream_data)", error);
    }
  }

  return NANOARROW_OK;
}

static void ArrowDeviceCudaRelease(struct ArrowDevice* device) {
  struct ArrowDeviceCudaPrivate* private_data =
      (struct ArrowDeviceCudaPrivate*)device->private_data;
  NANOARROW_CUDA_ASSERT_OK(cuDevicePrimaryCtxRelease(private_data->cu_device));
  ArrowFree(device->private_data);
  device->release = NULL;
}

static ArrowErrorCode ArrowDeviceCudaArrayViewCopy(struct ArrowDeviceArrayView* src,
                                                   struct ArrowDevice* device_dst,
                                                   struct ArrowDeviceArray* dst) {
  switch (src->device->device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
    case ARROW_DEVICE_CPU:
      switch (device_dst->device_type) {
        case ARROW_DEVICE_CUDA:
        case ARROW_DEVICE_CUDA_HOST:
        case ARROW_DEVICE_CPU:
          break;
        default:
          return ENOTSUP;
      }
      break;
    default:
      return ENOTSUP;
  }

  struct ArrowError error;

  struct ArrowArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(&tmp, &src->array_view, &error));

  int result =
      ArrowDeviceCudaArrayViewCopyInternal(src->device, src, device_dst, &tmp, &error);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  result = ArrowArrayFinishBuilding(&tmp, NANOARROW_VALIDATION_LEVEL_MINIMAL, &error);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  result = ArrowDeviceArrayInit(device_dst, dst, &tmp, NULL);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(&tmp);
    return result;
  }

  return result;
}

static ArrowErrorCode ArrowDeviceCudaInitDevice(struct ArrowDevice* device,
                                                ArrowDeviceType device_type,
                                                int64_t device_id,
                                                struct ArrowError* error) {
  switch (device_type) {
    case ARROW_DEVICE_CUDA:
    case ARROW_DEVICE_CUDA_HOST:
      break;
    default:
      ArrowErrorSet(error, "Device type code %" PRId32 " not supported", device_type);
      return EINVAL;
  }

  CUdevice cu_device;
  NANOARROW_CUDA_RETURN_NOT_OK(cuDeviceGet(&cu_device, device_id), "cuDeviceGet", error);

  CUcontext cu_context;
  NANOARROW_CUDA_RETURN_NOT_OK(cuDevicePrimaryCtxRetain(&cu_context, cu_device),
                               "cuDevicePrimaryCtxRetain", error);

  struct ArrowDeviceCudaPrivate* private_data =
      (struct ArrowDeviceCudaPrivate*)ArrowMalloc(sizeof(struct ArrowDeviceCudaPrivate));
  if (private_data == NULL) {
    NANOARROW_CUDA_ASSERT_OK(cuDevicePrimaryCtxRelease(cu_device));
    ArrowErrorSet(error, "out of memory");
    return ENOMEM;
  }

  device->device_type = device_type;
  device->device_id = device_id;
  device->array_init = &ArrowDeviceCudaArrayInit;
  device->array_move = &ArrowDeviceCudaArrayMove;
  device->array_copy = &ArrowDeviceCudaArrayViewCopy;
  device->buffer_init = &ArrowDeviceCudaBufferInit;
  device->buffer_move = NULL;
  device->buffer_copy = &ArrowDeviceCudaBufferCopy;
  device->synchronize_event = &ArrowDeviceCudaSynchronize;
  device->release = &ArrowDeviceCudaRelease;

  private_data->cu_device = cu_device;
  private_data->cu_context = cu_context;
  device->private_data = private_data;

  return NANOARROW_OK;
}

struct ArrowDevice* ArrowDeviceCuda(ArrowDeviceType device_type, int64_t device_id) {
  CUresult err;
  int n_devices;

  static struct ArrowDevice* devices_singleton = NULL;
  if (devices_singleton == NULL) {
    err = cuInit(0);
    if (err != CUDA_SUCCESS) {
      return NULL;
    }

    err = cuDeviceGetCount(&n_devices);
    if (err != CUDA_SUCCESS) {
      return NULL;
    }

    if (n_devices == 0) {
      return NULL;
    }

    devices_singleton =
        (struct ArrowDevice*)ArrowMalloc(2 * n_devices * sizeof(struct ArrowDevice));
    if (devices_singleton == NULL) {
      return NULL;
    }

    int result = NANOARROW_OK;
    memset(devices_singleton, 0, 2 * n_devices * sizeof(struct ArrowDevice));

    for (int i = 0; i < n_devices; i++) {
      result =
          ArrowDeviceCudaInitDevice(devices_singleton + i, ARROW_DEVICE_CUDA, i, NULL);
      if (result != NANOARROW_OK) {
        break;
      }

      result = ArrowDeviceCudaInitDevice(devices_singleton + n_devices + i,
                                         ARROW_DEVICE_CUDA_HOST, i, NULL);
      if (result != NANOARROW_OK) {
        break;
      }
    }

    if (result != NANOARROW_OK) {
      for (int i = 0; i < n_devices; i++) {
        if (devices_singleton[i].release != NULL) {
          devices_singleton[i].release(&(devices_singleton[i]));
        }
      }

      ArrowFree(devices_singleton);
      devices_singleton = NULL;
      return NULL;
    }

  } else {
    err = cuDeviceGetCount(&n_devices);
    if (err != CUDA_SUCCESS) {
      return NULL;
    }
  }

  if (device_id < 0 || device_id >= n_devices) {
    return NULL;
  }

  switch (device_type) {
    case ARROW_DEVICE_CUDA:
      return devices_singleton + device_id;
    case ARROW_DEVICE_CUDA_HOST:
      return devices_singleton + n_devices + device_id;
    default:
      return NULL;
  }
}
