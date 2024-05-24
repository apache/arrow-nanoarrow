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

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "nanoarrow_device.h"
#include "nanoarrow_types.h"

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
    const CUresult NAME = (EXPR);                                 \
    if (NAME != CUDA_SUCCESS) {                                   \
      ArrowDeviceCudaSetError(NAME, OP, ERROR);                   \
      return EIO;                                                 \
    }                                                             \
  } while (0)

#define NANOARROW_CUDA_RETURN_NOT_OK(EXPR, OP, ERROR)                                    \
  _NANOARROW_CUDA_RETURN_NOT_OK_IMPL(_NANOARROW_MAKE_NAME(cuda_err_, __COUNTER__), EXPR, \
                                     OP, ERROR)

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
      cuMemFreeHost(allocator_private->allocated_ptr);
      break;
    default:
      break;
  }

  ArrowFree(allocator_private);
}

static ArrowErrorCode ArrowDeviceCudaAllocateBuffer(struct ArrowDevice* device,
                                                    struct ArrowBuffer* buffer,
                                                    int64_t size_bytes) {
  int prev_device = 0;
  cudaError_t result = cudaGetDevice(&prev_device);
  if (result != cudaSuccess) {
    return EINVAL;
  }

  result = cudaSetDevice((int)device->device_id);
  if (result != cudaSuccess) {
    cudaSetDevice(prev_device);
    return EINVAL;
  }

  struct ArrowDeviceCudaAllocatorPrivate* allocator_private =
      (struct ArrowDeviceCudaAllocatorPrivate*)ArrowMalloc(
          sizeof(struct ArrowDeviceCudaAllocatorPrivate));
  if (allocator_private == NULL) {
    cudaSetDevice(prev_device);
    return ENOMEM;
  }

  void* ptr = NULL;
  switch (device->device_type) {
    case ARROW_DEVICE_CUDA:
      result = cudaMalloc(&ptr, (int64_t)size_bytes);
      break;
    case ARROW_DEVICE_CUDA_HOST:
      result = cudaMallocHost(&ptr, (int64_t)size_bytes);
      break;
    default:
      ArrowFree(allocator_private);
      cudaSetDevice(prev_device);
      return EINVAL;
  }

  if (result != cudaSuccess) {
    ArrowFree(allocator_private);
    cudaSetDevice(prev_device);
    return ENOMEM;
  }

  allocator_private->device_id = device->device_id;
  allocator_private->device_type = device->device_type;
  allocator_private->allocated_ptr = ptr;

  buffer->data = (uint8_t*)ptr;
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = size_bytes;
  buffer->allocator =
      ArrowBufferDeallocator(&ArrowDeviceCudaDeallocator, allocator_private);

  cudaSetDevice(prev_device);
  return NANOARROW_OK;
}

struct ArrowDeviceCudaArrayPrivate {
  struct ArrowArray parent;
  cudaEvent_t sync_event;
};

static void ArrowDeviceCudaArrayRelease(struct ArrowArray* array) {
  struct ArrowDeviceCudaArrayPrivate* private_data =
      (struct ArrowDeviceCudaArrayPrivate*)array->private_data;
  cudaEventDestroy(private_data->sync_event);
  ArrowArrayRelease(&private_data->parent);
  ArrowFree(private_data);
  array->release = NULL;
}

static ArrowErrorCode ArrowDeviceCudaArrayInit(struct ArrowDevice* device,
                                               struct ArrowDeviceArray* device_array,
                                               struct ArrowArray* array) {
  struct ArrowDeviceCudaArrayPrivate* private_data =
      (struct ArrowDeviceCudaArrayPrivate*)ArrowMalloc(
          sizeof(struct ArrowDeviceCudaArrayPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  int prev_device = 0;
  cudaError_t result = cudaGetDevice(&prev_device);
  if (result != cudaSuccess) {
    ArrowFree(private_data);
    return EINVAL;
  }

  result = cudaSetDevice((int)device->device_id);
  if (result != cudaSuccess) {
    cudaSetDevice(prev_device);
    ArrowFree(private_data);
    return EINVAL;
  }

  cudaError_t error = cudaEventCreate(&private_data->sync_event);
  if (error != cudaSuccess) {
    ArrowFree(private_data);
    return EINVAL;
  }

  memset(device_array, 0, sizeof(struct ArrowDeviceArray));
  device_array->array = *array;
  device_array->array.private_data = private_data;
  device_array->array.release = &ArrowDeviceCudaArrayRelease;
  ArrowArrayMove(array, &private_data->parent);

  device_array->device_id = device->device_id;
  device_array->device_type = device->device_type;
  device_array->sync_event = &private_data->sync_event;

  cudaSetDevice(prev_device);
  return NANOARROW_OK;
}

// TODO: All these buffer copiers would benefit from cudaMemcpyAsync but there is
// no good way to incorporate that just yet

static ArrowErrorCode ArrowDeviceCudaBufferInit(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBuffer* dst) {
  struct ArrowBuffer tmp;
  enum cudaMemcpyKind memcpy_kind;

  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaAllocateBuffer(device_dst, &tmp, src.size_bytes));
    memcpy_kind = cudaMemcpyHostToDevice;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaAllocateBuffer(device_dst, &tmp, src.size_bytes));
    memcpy_kind = cudaMemcpyDeviceToDevice;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    ArrowBufferInit(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&tmp, src.size_bytes));
    tmp.size_bytes = src.size_bytes;
    memcpy_kind = cudaMemcpyDeviceToHost;

  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaAllocateBuffer(device_dst, &tmp, src.size_bytes));
    memcpy_kind = cudaMemcpyHostToHost;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    NANOARROW_RETURN_NOT_OK(
        ArrowDeviceCudaAllocateBuffer(device_dst, &tmp, src.size_bytes));
    memcpy_kind = cudaMemcpyHostToHost;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    ArrowBufferInit(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&tmp, src.size_bytes));
    tmp.size_bytes = src.size_bytes;
    memcpy_kind = cudaMemcpyHostToHost;

  } else {
    return ENOTSUP;
  }

  cudaError_t result =
      cudaMemcpy(tmp.data, src.data.as_uint8, (size_t)src.size_bytes, memcpy_kind);
  if (result != cudaSuccess) {
    ArrowBufferReset(&tmp);
    return EINVAL;
  }

  ArrowBufferMove(&tmp, dst);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowDeviceCudaBufferCopy(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBufferView dst) {
  enum cudaMemcpyKind memcpy_kind;

  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    memcpy_kind = cudaMemcpyHostToDevice;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA) {
    memcpy_kind = cudaMemcpyDeviceToDevice;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy_kind = cudaMemcpyDeviceToHost;
  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy_kind = cudaMemcpyHostToHost;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy_kind = cudaMemcpyHostToHost;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy_kind = cudaMemcpyHostToHost;
  } else {
    return ENOTSUP;
  }

  cudaError_t result = cudaMemcpy((void*)dst.data.as_uint8, src.data.as_uint8,
                                  dst.size_bytes, memcpy_kind);
  if (result != cudaSuccess) {
    return EINVAL;
  }
  return NANOARROW_OK;
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
  cudaEvent_t* cuda_event = (cudaEvent_t*)sync_event;
  cudaError_t result = cudaEventSynchronize(*cuda_event);

  if (result != cudaSuccess) {
    ArrowErrorSet(error, "cudaEventSynchronize() failed: %s", cudaGetErrorString(result));
    return EINVAL;
  }

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

static void ArrowDeviceCudaRelease(struct ArrowDevice* device) {
  struct ArrowDeviceCudaPrivate* private_data =
      (struct ArrowDeviceCudaPrivate*)device->private_data;
  cuDevicePrimaryCtxRelease(private_data->cu_device);
  ArrowFree(device->private_data);
  device->release = NULL;
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
      ArrowErrorSet(error, "Device type code %d not supported", (int)device_type);
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
    cuDevicePrimaryCtxRelease(cu_device);
    ArrowErrorSet(error, "out of memory");
    return ENOMEM;
  }

  device->device_type = device_type;
  device->device_id = device_id;
  device->array_init = &ArrowDeviceCudaArrayInit;
  device->array_move = &ArrowDeviceCudaArrayMove;
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
  int n_devices;
  cudaError_t result = cudaGetDeviceCount(&n_devices);
  if (result != cudaSuccess) {
    return NULL;
  }
  static struct ArrowDevice* devices_singleton = NULL;
  if (devices_singleton == NULL) {
    devices_singleton =
        (struct ArrowDevice*)ArrowMalloc(2 * n_devices * sizeof(struct ArrowDevice));

    for (int i = 0; i < n_devices; i++) {
      int result =
          ArrowDeviceCudaInitDevice(devices_singleton + i, ARROW_DEVICE_CUDA, i, NULL);
      if (result != NANOARROW_OK) {
        ArrowFree(devices_singleton);
        devices_singleton = NULL;
      }

      result = ArrowDeviceCudaInitDevice(devices_singleton + n_devices + i,
                                         ARROW_DEVICE_CUDA_HOST, i, NULL);
      if (result != NANOARROW_OK) {
        ArrowFree(devices_singleton);
        devices_singleton = NULL;
      }
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
