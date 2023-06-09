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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "nanoarrow_device.h"
#include "nanoarrow_device_cuda.h"

TEST(NanoarrowDeviceCuda, GetDevice) {
  struct ArrowDevice* cuda = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  ASSERT_NE(cuda, nullptr);
  EXPECT_EQ(cuda->device_type, ARROW_DEVICE_CUDA);
  struct ArrowDevice* cuda_host = ArrowDeviceCuda(ARROW_DEVICE_CUDA_HOST, 0);
  ASSERT_NE(cuda_host, nullptr);
  EXPECT_EQ(cuda_host->device_type, ARROW_DEVICE_CUDA_HOST);

  // null return for invalid input
  EXPECT_EQ(ArrowDeviceCuda(ARROW_DEVICE_CUDA, std::numeric_limits<int32_t>::max()),
            nullptr);
  EXPECT_EQ(ArrowDeviceCuda(ARROW_DEVICE_CPU, 0), nullptr);
}

TEST(NanoarrowDeviceCuda, DeviceCudaBufferInit) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  struct ArrowBuffer buffer_gpu;
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_gpu), NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  // (Content is tested on the roundtrip)
  struct ArrowDeviceBufferView gpu_view = {buffer_gpu.data, 0, buffer_gpu.size_bytes};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, gpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  // (Content is tested on the roundtrip)
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, cpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  ArrowBufferReset(&buffer_gpu);
}

TEST(NanoarrowDeviceCuda, DeviceCudaHostBufferInit) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA_HOST, 0);
  struct ArrowBuffer buffer_gpu;
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_gpu), NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer_gpu.data, data, sizeof(data)), 0);
  // Here, "GPU" is memory in the CPU space allocated by cudaMallocHost
  struct ArrowDeviceBufferView gpu_view = {buffer_gpu.data, 0, buffer_gpu.size_bytes};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, gpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, cpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  ArrowBufferReset(&buffer_gpu);
}

TEST(NanoarrowDeviceCuda, DeviceCudaBufferCopy) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};

  void* gpu_dest;
  cudaError_t result = cudaMalloc(&gpu_dest, sizeof(data));
  struct ArrowDeviceBufferView gpu_view = {gpu_dest, 0, sizeof(data)};
  if (result != cudaSuccess) {
    GTEST_FAIL() << "cudaMalloc(&gpu_dest) failed";
  }

  void* gpu_dest2;
  result = cudaMalloc(&gpu_dest2, sizeof(data));
  struct ArrowDeviceBufferView gpu_view2 = {gpu_dest2, 0, sizeof(data)};
  if (result != cudaSuccess) {
    GTEST_FAIL() << "cudaMalloc(&gpu_dest2) failed";
  }

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, cpu_view, gpu, gpu_view), NANOARROW_OK);

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, gpu, gpu_view2), NANOARROW_OK);

  // GPU -> CPU
  uint8_t cpu_dest[5];
  struct ArrowDeviceBufferView cpu_dest_view = {cpu_dest, 0, sizeof(data)};
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, cpu, cpu_dest_view), NANOARROW_OK);

  // Check roundtrip
  EXPECT_EQ(memcmp(cpu_dest, data, sizeof(data)), 0);

  // Clean up
  result = cudaFree(gpu_dest);
  if (result != cudaSuccess) {
    GTEST_FAIL() << "cudaFree(gpu_dest) failed";
  }

  result = cudaFree(gpu_dest2);
  if (result != cudaSuccess) {
    GTEST_FAIL() << "cudaFree(gpu_dest2) failed";
  }
}
