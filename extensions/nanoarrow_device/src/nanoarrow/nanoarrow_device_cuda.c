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



static ArrowErrorCode ArrowDeviceCudaSynchronize(struct ArrowDevice* device,
                                                 struct ArrowDevice* device_event,
                                                 void* sync_event,
                                                 struct ArrowError* error) {
  if (sync_event == NULL) {
    return NANOARROW_OK;
  }

  if (device_event->device_type != ARROW_DEVICE_CUDA ||
      device_event->device_type != ARROW_DEVICE_CUDA_HOST) {
    return ENOTSUP;
  }

  cudaEvent_t* cuda_event = (cudaEvent_t*)sync_event;
  cudaError_t result = cudaEventSynchronize(*cuda_event);

  if (result != cudaSuccess) {
    ArrowErrorSet(error, "cudaEventSynchronize() failed: %s", cudaGetErrorString(result));
    return EINVAL;
  }

  cudaEventDestroy(*cuda_event);
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
  device->buffer_init = NULL;
  device->buffer_move = NULL;
  device->buffer_copy = NULL;
  device->copy_required = NULL;
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

      result = ArrowDeviceCudaInitDevice(devices_singleton + (2 * i), ARROW_DEVICE_CUDA,
                                         i, NULL);
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
      return devices_singleton + (2 * device_id);
    default:
      return NULL;
  }
}
