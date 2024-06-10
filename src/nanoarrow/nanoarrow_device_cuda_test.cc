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

#include <cuda.h>
#include <gtest/gtest.h>

#include "nanoarrow_device.h"
#include "nanoarrow_device_cuda.h"

class CudaTemporaryContext {
 public:
  CudaTemporaryContext(int device_id) : initialized_(false) {
    CUresult err = cuDeviceGet(&device_, device_id);
    if (err != CUDA_SUCCESS) {
      return;
    }

    err = cuDevicePrimaryCtxRetain(&context_, device_);
    if (err != CUDA_SUCCESS) {
      return;
    }

    cuCtxPushCurrent(context_);
    initialized_ = true;
  }

  bool valid() { return initialized_; }

  ~CudaTemporaryContext() {
    if (initialized_) {
      CUcontext unused;
      cuCtxPopCurrent(&unused);
      cuDevicePrimaryCtxRelease(device_);
    }
  }

 private:
  bool initialized_;
  CUdevice device_;
  CUcontext context_;
};

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
  ASSERT_NE(gpu, nullptr);

  struct ArrowBuffer buffer_gpu;
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView cpu_view = {data, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_gpu), NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  // (Content is tested on the roundtrip)
  struct ArrowBufferView gpu_view = {buffer_gpu.data, buffer_gpu.size_bytes};

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
  ASSERT_NE(gpu, nullptr);

  struct ArrowBuffer buffer_gpu;
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView cpu_view = {data, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_gpu), NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer_gpu.data, data, sizeof(data)), 0);
  // Here, "GPU" is memory in the CPU space allocated by cudaMallocHost
  struct ArrowBufferView gpu_view = {buffer_gpu.data, buffer_gpu.size_bytes};

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
  ASSERT_NE(gpu, nullptr);

  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView cpu_view = {data, sizeof(data)};

  CudaTemporaryContext ctx(0);
  ASSERT_TRUE(ctx.valid());

  CUdeviceptr gpu_dest;
  CUresult result = cuMemAlloc(&gpu_dest, sizeof(data));
  struct ArrowBufferView gpu_view = {reinterpret_cast<void*>(gpu_dest), sizeof(data)};
  if (result != CUDA_SUCCESS) {
    GTEST_FAIL() << "cuMemAlloc() failed";
  }

  CUdeviceptr gpu_dest2;
  result = cuMemAlloc(&gpu_dest2, sizeof(data));
  struct ArrowBufferView gpu_view2 = {reinterpret_cast<void*>(gpu_dest), sizeof(data)};
  if (result != CUDA_SUCCESS) {
    GTEST_FAIL() << "cuMemAlloc() failed";
  }

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, cpu_view, gpu, gpu_view), NANOARROW_OK);

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, gpu, gpu_view2), NANOARROW_OK);

  // GPU -> CPU
  uint8_t cpu_dest[5];
  struct ArrowBufferView cpu_dest_view = {cpu_dest, sizeof(data)};
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, cpu, cpu_dest_view), NANOARROW_OK);

  // Check roundtrip
  EXPECT_EQ(memcmp(cpu_dest, data, sizeof(data)), 0);

  // Clean up
  result = cuMemFree(gpu_dest);
  if (result != CUDA_SUCCESS) {
    GTEST_FAIL() << "cuMemFree() failed";
  }

  result = cuMemFree(gpu_dest2);
  if (result != CUDA_SUCCESS) {
    GTEST_FAIL() << "cuMemFree() failed";
  }
}

class StringTypeParameterizedTestFixture
    : public ::testing::TestWithParam<std::pair<ArrowDeviceType, enum ArrowType>> {
 protected:
  std::pair<ArrowDeviceType, enum ArrowType> info;
};

std::pair<ArrowDeviceType, enum ArrowType> DeviceAndType(ArrowDeviceType device_type,
                                                         enum ArrowType arrow_type) {
  return {device_type, arrow_type};
}

TEST_P(StringTypeParameterizedTestFixture, ArrowDeviceCudaArrayViewString) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(GetParam().first, 0);
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType string_type = GetParam().second;

  ASSERT_EQ(ArrowArrayInitFromType(&array, string_type), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("abc")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("defg")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array, nullptr), NANOARROW_OK);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(device_array.array.length, 3);

  // Copy required to Cuda
  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array, gpu, &device_array2), ENOTSUP);
  ASSERT_EQ(ArrowDeviceArrayViewCopy(&device_array_view, gpu, &device_array2),
            NANOARROW_OK);
  ArrowArrayRelease(&device_array.array);

  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, gpu->device_id);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array2, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(device_array_view.array_view.length, 3);
  EXPECT_EQ(device_array2.array.length, 3);

  // Copy required back to Cpu for Cuda; not for CudaHost
  if (gpu->device_type == ARROW_DEVICE_CUDA_HOST) {
    ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array2, cpu, &device_array),
              NANOARROW_OK);
  } else {
    ASSERT_EQ(ArrowDeviceArrayViewCopy(&device_array_view, cpu, &device_array),
              NANOARROW_OK);
    ArrowArrayRelease(&device_array2.array);
  }

  ASSERT_NE(device_array.array.release, nullptr);
  ASSERT_EQ(device_array.device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefg", 7),
            0);

  ArrowArrayRelease(&device_array.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowDeviceCuda, StringTypeParameterizedTestFixture,
    ::testing::Values(DeviceAndType(ARROW_DEVICE_CUDA, NANOARROW_TYPE_STRING),
                      DeviceAndType(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_STRING),
                      DeviceAndType(ARROW_DEVICE_CUDA, NANOARROW_TYPE_BINARY),
                      DeviceAndType(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_BINARY),
                      DeviceAndType(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_STRING),
                      DeviceAndType(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_LARGE_STRING),
                      DeviceAndType(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_BINARY),
                      DeviceAndType(ARROW_DEVICE_CUDA_HOST,
                                    NANOARROW_TYPE_LARGE_BINARY)));
