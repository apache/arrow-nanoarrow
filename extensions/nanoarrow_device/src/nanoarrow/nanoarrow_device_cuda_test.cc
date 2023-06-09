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

  ArrowDeviceArrayInit(&device_array, cpu);
  ArrowArrayMove(&array, &device_array.array);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(device_array.array.length, 3);

  // Copy required to Cuda
  ASSERT_TRUE(ArrowDeviceArrayViewCopyRequired(&device_array_view, gpu));

  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array, &device_array_view, gpu, &device_array2),
      NANOARROW_OK);
  ASSERT_EQ(device_array.array.release, nullptr);
  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, gpu->device_id);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array2, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(device_array_view.array_view.length, 3);
  EXPECT_EQ(device_array2.array.length, 3);

  // Copy shouldn't be required to the same device
  ASSERT_FALSE(ArrowDeviceArrayViewCopyRequired(&device_array_view, gpu));

  // Copy required back to Cpu for Cuda; not for CudaHost
  ASSERT_EQ(ArrowDeviceArrayViewCopyRequired(&device_array_view, cpu),
            gpu->device_type == ARROW_DEVICE_CUDA);
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array2, &device_array_view, cpu, &device_array),
      NANOARROW_OK);
  ASSERT_EQ(device_array2.array.release, nullptr);
  ASSERT_NE(device_array.array.release, nullptr);
  ASSERT_EQ(device_array.device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefg", 7),
            0);

  device_array.array.release(&device_array.array);
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

class ListTypeParameterizedTestFixture : public ::testing::TestWithParam<enum ArrowType> {
 protected:
  enum ArrowType type;
};

TEST_P(ListTypeParameterizedTestFixture, ArrowDeviceCudaArrayViewList) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);
  struct ArrowSchema schema;
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType list_type = GetParam();

  ASSERT_EQ(ArrowSchemaInitFromType(&schema, list_type), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetType(schema.children[0], NANOARROW_TYPE_INT32), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(&array, &schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 123), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 456), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(array.children[0], 789), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(&array), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ArrowDeviceArrayInit(&device_array, cpu);
  ArrowArrayMove(&array, &device_array.array);

  ArrowDeviceArrayViewInit(&device_array_view);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(&device_array_view.array_view, &schema, nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.children[0]->buffer_views[1].size_bytes,
            3 * sizeof(int32_t));

  // Copy required to Cuda
  ASSERT_TRUE(ArrowDeviceArrayViewCopyRequired(&device_array_view, gpu));

  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array, &device_array_view, gpu, &device_array2),
      NANOARROW_OK);
  ASSERT_EQ(device_array.array.release, nullptr);
  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, gpu->device_id);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array2, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.children[0]->buffer_views[1].size_bytes,
            3 * sizeof(int32_t));

  // Copy shouldn't be required to the same device
  ASSERT_FALSE(ArrowDeviceArrayViewCopyRequired(&device_array_view, gpu));

  // Copy required back to the CPU if Cuda, not for CudaHost
  ASSERT_EQ(ArrowDeviceArrayViewCopyRequired(&device_array_view, cpu),
            gpu->device_type == ARROW_DEVICE_CUDA);
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array2, &device_array_view, cpu, &device_array),
      NANOARROW_OK);
  ASSERT_EQ(device_array2.array.release, nullptr);
  ASSERT_NE(device_array.array.release, nullptr);
  ASSERT_EQ(device_array.device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.length, 3);
  EXPECT_EQ(device_array_view.array_view.null_count, 1);

  struct ArrowBufferView data_view =
      device_array_view.array_view.children[0]->buffer_views[1];
  ASSERT_EQ(data_view.size_bytes, 3 * sizeof(int32_t));
  EXPECT_EQ(data_view.data.as_int32[0], 123);
  EXPECT_EQ(data_view.data.as_int32[1], 456);
  EXPECT_EQ(data_view.data.as_int32[2], 789);

  schema.release(&schema);
  device_array.array.release(&device_array.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowDeviceCuda, ListTypeParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_LIST,
                                           NANOARROW_TYPE_LARGE_LIST));
