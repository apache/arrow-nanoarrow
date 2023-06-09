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

#include <gtest/gtest.h>
#include "nanoarrow_device.h"

#include "nanoarrow_device_cuda.h"

TEST(NanoarrowDeviceCuda, GetDevice) {
  struct ArrowDevice* cuda = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  ASSERT_NE(cuda, nullptr);
  struct ArrowDevice* cuda_host = ArrowDeviceCuda(ARROW_DEVICE_CUDA_HOST, 0);
  ASSERT_NE(cuda_host, nullptr);
}


TEST(NanoarrowDeviceCuda, DeviceGpuBufferInit) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  struct ArrowBuffer buffer_gpu;
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_gpu), NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  // EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  struct ArrowDeviceBufferView gpu_view = {buffer_gpu.data, 0, buffer_gpu.size_bytes};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, gpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  // EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, cpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  ArrowBufferReset(&buffer_gpu);
}
