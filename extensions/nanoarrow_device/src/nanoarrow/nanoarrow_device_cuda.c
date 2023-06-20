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

#include <cuda_runtime_api.h>

#include "nanoarrow_device.h"

static void ArrowDeviceCudaAllocatorFree(struct ArrowBufferAllocator* allocator,
                                         uint8_t* ptr, int64_t old_size) {
  if (ptr != NULL) {
    // TODO: We need the device_id here
    cudaFree(ptr);
  }
}

static uint8_t* ArrowDeviceCudaAllocatorReallocate(struct ArrowBufferAllocator* allocator,
                                                   uint8_t* ptr, int64_t old_size,
                                                   int64_t new_size) {
  ArrowDeviceCudaAllocatorFree(allocator, ptr, old_size);
  return NULL;
}

static ArrowErrorCode ArrowDeviceCudaAllocateBuffer(struct ArrowBuffer* buffer,
                                                    int64_t size_bytes) {
  void* ptr = NULL;
  cudaError_t result = cudaMalloc(&ptr, (int64_t)size_bytes);
  if (result != cudaSuccess) {
    return EINVAL;
  }

  buffer->data = (uint8_t*)ptr;
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = size_bytes;
  buffer->allocator.reallocate = &ArrowDeviceCudaAllocatorReallocate;
  buffer->allocator.free = &ArrowDeviceCudaAllocatorFree;
  // TODO: We need the device_id here
  buffer->allocator.private_data = NULL;
  return NANOARROW_OK;
}

static void ArrowDeviceCudaHostAllocatorFree(struct ArrowBufferAllocator* allocator,
                                             uint8_t* ptr, int64_t old_size) {
  if (ptr != NULL) {
    // TODO: We need the device_id here
    cudaFreeHost(ptr);
  }
}

static uint8_t* ArrowDeviceCudaHostAllocatorReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  ArrowDeviceCudaHostAllocatorFree(allocator, ptr, old_size);
  return NULL;
}

static ArrowErrorCode ArrowDeviceCudaHostAllocateBuffer(struct ArrowBuffer* buffer,
                                                        int64_t size_bytes) {
  void* ptr = NULL;
  cudaError_t result = cudaMallocHost(&ptr, (int64_t)size_bytes);
  if (result != cudaSuccess) {
    return EINVAL;
  }

  buffer->data = (uint8_t*)ptr;
  buffer->size_bytes = size_bytes;
  buffer->capacity_bytes = size_bytes;
  buffer->allocator.reallocate = &ArrowDeviceCudaHostAllocatorReallocate;
  buffer->allocator.free = &ArrowDeviceCudaHostAllocatorFree;
  // TODO: We need the device_id here
  buffer->allocator.private_data = NULL;
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
  private_data->parent.release(&private_data->parent);
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

  // TODO: Set device
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

  return NANOARROW_OK;
}

// TODO: All these buffer copiers would benefit from cudaMemcpyAsync but there is
// no good way to incorporate that just yet

static ArrowErrorCode ArrowDeviceCudaBufferInit(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBuffer* dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    struct ArrowBuffer tmp;
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaAllocateBuffer(&tmp, src.size_bytes));
    cudaError_t result = cudaMemcpy(tmp.data, src.data.as_uint8, (size_t)src.size_bytes,
                                    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      ArrowBufferReset(&tmp);
      return EINVAL;
    }

    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA) {
    struct ArrowBuffer tmp;
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaAllocateBuffer(&tmp, src.size_bytes));
    cudaError_t result = cudaMemcpy(tmp.data, src.data.as_uint8, (size_t)src.size_bytes,
                                    cudaMemcpyDeviceToDevice);
    if (result != cudaSuccess) {
      ArrowBufferReset(&tmp);
      return EINVAL;
    }

    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    struct ArrowBuffer tmp;
    ArrowBufferInit(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&tmp, src.size_bytes));
    tmp.size_bytes = src.size_bytes;
    cudaError_t result = cudaMemcpy(tmp.data, src.data.as_uint8, (size_t)src.size_bytes,
                                    cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      ArrowBufferReset(&tmp);
      return EINVAL;
    }

    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaHostAllocateBuffer(dst, src.size_bytes));
    memcpy(dst->data, src.data.as_uint8, (size_t)src.size_bytes);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    NANOARROW_RETURN_NOT_OK(ArrowDeviceCudaHostAllocateBuffer(dst, src.size_bytes));
    memcpy(dst->data, src.data.as_uint8, (size_t)src.size_bytes);
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    struct ArrowBuffer tmp;
    ArrowBufferInit(&tmp);
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(&tmp, src.size_bytes));
    tmp.size_bytes = src.size_bytes;
    memcpy(tmp.data, src.data.as_uint8, (size_t)src.size_bytes);
    ArrowBufferMove(&tmp, dst);
    return NANOARROW_OK;

  } else {
    return ENOTSUP;
  }
}

static ArrowErrorCode ArrowDeviceCudaBufferCopy(struct ArrowDevice* device_src,
                                                struct ArrowBufferView src,
                                                struct ArrowDevice* device_dst,
                                                struct ArrowBufferView dst) {
  // This is all just cudaMemcpy or memcpy
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    cudaError_t result = cudaMemcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes,
                                    cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      return EINVAL;
    }
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA) {
    cudaError_t result = cudaMemcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes,
                                    cudaMemcpyDeviceToDevice);
    if (result != cudaSuccess) {
      return EINVAL;
    }
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    cudaError_t result = cudaMemcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes,
                                    cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      return EINVAL;
    }
    return NANOARROW_OK;

  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    memcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    memcpy(dst.data.as_uint8, src.data.as_uint8, dst.size_bytes);
    return NANOARROW_OK;
  } else {
    return ENOTSUP;
  }
}

static int ArrowDeviceCudaCopyRequired(struct ArrowDevice* device_src,
                                       struct ArrowArrayView* src,
                                       struct ArrowDevice* device_dst) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
      device_dst->device_type == ARROW_DEVICE_CUDA) {
    // Copy
    return 1;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CUDA &&
             device_src->device_id == device_dst->device_id) {
    // Move
    return 0;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    // Copy
    return 1;

  } else if (device_src->device_type == ARROW_DEVICE_CPU &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST) {
    // Copy: we can't assume the memory has been registered. A user can force
    // this by registering the memory and setting device->device_type manually.
    // A copy will ensure all buffers are allocated with cudaMallocHost().
    return 1;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_src->device_id == device_dst->device_id) {
    // Move
    return 0;

  } else if (device_src->device_type == ARROW_DEVICE_CUDA_HOST &&
             device_dst->device_type == ARROW_DEVICE_CPU) {
    // Move: the array's release callback is responsible for cudaFreeHost or
    // deregistration (or perhaps this has been handled at a higher level)
    return 0;

  } else {
    // Fall back to the other device's implementation
    return -1;
  }
}

static ArrowErrorCode ArrowDeviceCudaSynchronize(struct ArrowDevice* device,
                                                 void* sync_event,
                                                 struct ArrowError* error) {
  if (sync_event == NULL) {
    return NANOARROW_OK;
  }

  if (device->device_type != ARROW_DEVICE_CUDA ||
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

static void ArrowDeviceCudaRelease(struct ArrowDevice* device) {
  // No private_data to release
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

  int n_devices;
  cudaError_t result = cudaGetDeviceCount(&n_devices);
  if (result != cudaSuccess) {
    ArrowErrorSet(error, "cudaGetDeviceCount() failed: %s", cudaGetErrorString(result));
    return EINVAL;
  }

  if (device_id < 0 || device_id >= n_devices) {
    ArrowErrorSet(error, "CUDA device_id must be between 0 and %d", n_devices - 1);
    return EINVAL;
  }

  device->device_type = device_type;
  device->device_id = device_id;
  device->array_init = &ArrowDeviceCudaArrayInit;
  device->buffer_init = &ArrowDeviceCudaBufferInit;
  device->buffer_move = NULL;
  device->buffer_copy = &ArrowDeviceCudaBufferCopy;
  device->copy_required = &ArrowDeviceCudaCopyRequired;
  device->synchronize_event = &ArrowDeviceCudaSynchronize;
  device->release = &ArrowDeviceCudaRelease;
  device->private_data = NULL;

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
