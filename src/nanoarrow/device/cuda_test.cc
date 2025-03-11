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
#include <errno.h>
#include <gtest/gtest.h>
#include <tuple>

#include "nanoarrow/nanoarrow_device.hpp"

class CudaTemporaryContext {
 public:
  CudaTemporaryContext(int64_t device_id) : initialized_(false) {
    CUresult err = cuDeviceGet(&device_, static_cast<int>(device_id));
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

class CudaStream {
 public:
  CudaStream(int64_t device_id) : device_id_(device_id), hstream_(0) {}

  ArrowErrorCode Init() {
    CudaTemporaryContext ctx(device_id_);
    if (!ctx.valid()) {
      return EINVAL;
    }

    if (cuStreamCreate(&hstream_, CU_STREAM_DEFAULT) != CUDA_SUCCESS) {
      return EINVAL;
    }

    return NANOARROW_OK;
  }

  CUstream* get() { return &hstream_; }

  ~CudaStream() {
    if (hstream_ != 0) {
      cuStreamDestroy(hstream_);
    }
  }

  int64_t device_id_;
  CUstream hstream_;
};

class CudaEvent {
 public:
  CudaEvent(int64_t device_id) : device_id_(device_id), hevent_(nullptr) {}

  ArrowErrorCode Init() {
    CudaTemporaryContext ctx(device_id_);
    if (!ctx.valid()) {
      return EINVAL;
    }

    if (cuEventCreate(&hevent_, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
      return EINVAL;
    }

    return NANOARROW_OK;
  }

  CUevent* get() { return &hevent_; }

  void release() { hevent_ = nullptr; }

  ~CudaEvent() {
    if (hevent_ != nullptr) {
      cuEventDestroy(hevent_);
    }
  }

  int64_t device_id_;
  CUevent hevent_;
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

  CudaStream stream(gpu->device_id);
  ASSERT_EQ(stream.Init(), NANOARROW_OK);

  // Failing to provide a stream should error
  ASSERT_EQ(ArrowDeviceBufferInitAsync(cpu, cpu_view, gpu, nullptr, nullptr), EINVAL);

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(cpu, cpu_view, gpu, &buffer_gpu, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  // (Content is tested on the roundtrip)
  struct ArrowBufferView gpu_view = {buffer_gpu.data, buffer_gpu.size_bytes};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(gpu, gpu_view, gpu, &buffer, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  // (Content is tested on the roundtrip)
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(gpu, gpu_view, cpu, &buffer, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));

  ASSERT_EQ(cuStreamSynchronize(*stream.get()), CUDA_SUCCESS);
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

  CudaStream stream(gpu->device_id);
  ASSERT_EQ(stream.Init(), NANOARROW_OK);

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(cpu, cpu_view, gpu, &buffer_gpu, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer_gpu.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer_gpu.data, data, sizeof(data)), 0);
  // Here, "GPU" is memory in the CPU space allocated by cudaMallocHost
  struct ArrowBufferView gpu_view = {buffer_gpu.data, buffer_gpu.size_bytes};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(gpu, gpu_view, gpu, &buffer, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInitAsync(gpu, gpu_view, cpu, &buffer, stream.get()),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));

  ASSERT_EQ(cuStreamSynchronize(*stream.get()), CUDA_SUCCESS);
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

  CudaStream stream(gpu->device_id);
  ASSERT_EQ(stream.Init(), NANOARROW_OK);

  // Failing to provide a stream should error
  ASSERT_EQ(ArrowDeviceBufferCopyAsync(cpu, cpu_view, gpu, gpu_view, nullptr), EINVAL);

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopyAsync(cpu, cpu_view, gpu, gpu_view, stream.get()),
            NANOARROW_OK);

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopyAsync(gpu, gpu_view, gpu, gpu_view2, stream.get()),
            NANOARROW_OK);

  // GPU -> CPU
  uint8_t cpu_dest[5];
  struct ArrowBufferView cpu_dest_view = {cpu_dest, sizeof(data)};
  ASSERT_EQ(ArrowDeviceBufferCopyAsync(gpu, gpu_view, cpu, cpu_dest_view, stream.get()),
            NANOARROW_OK);

  // Check roundtrip
  ASSERT_EQ(cuStreamSynchronize(*stream.get()), CUDA_SUCCESS);
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

TEST(NanoarrowDeviceCuda, DeviceCudaArrayInit) {
  struct ArrowDevice* gpu = ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0);

  CudaStream stream(gpu->device_id);
  ASSERT_EQ(stream.Init(), NANOARROW_OK);

  CudaEvent event(gpu->device_id);
  ASSERT_EQ(event.Init(), NANOARROW_OK);

  struct ArrowDeviceArray device_array;
  struct ArrowArray array;
  array.release = nullptr;

  // No provided sync event should result in a null sync event in the final array
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowDeviceArrayInit(gpu, &device_array, &array, nullptr), NANOARROW_OK);
  ASSERT_EQ(device_array.sync_event, nullptr);
  ArrowArrayRelease(&device_array.array);

  // Provided sync event should result in ownership of the event being taken by the
  // device array.
  device_array.sync_event = nullptr;
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowDeviceArrayInit(gpu, &device_array, &array, event.get()), NANOARROW_OK);
  ASSERT_EQ(*((CUevent*)device_array.sync_event), *event.get());
  event.release();
  ArrowArrayRelease(&device_array.array);

  // Provided stream without provided event should result in an event created by and owned
  // by the device array
  device_array.sync_event = nullptr;
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowDeviceArrayInitAsync(gpu, &device_array, &array, nullptr, stream.get()),
            NANOARROW_OK);
  ASSERT_NE(*(CUevent*)device_array.sync_event, nullptr);
  ArrowArrayRelease(&device_array.array);

  // Provided stream and sync event should result in the device array taking ownership
  // and recording the event
  ASSERT_EQ(event.Init(), NANOARROW_OK);
  device_array.sync_event = nullptr;
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(
      ArrowDeviceArrayInitAsync(gpu, &device_array, &array, event.get(), stream.get()),
      NANOARROW_OK);
  ASSERT_EQ(*((CUevent*)device_array.sync_event), *event.get());
  event.release();
  ArrowArrayRelease(&device_array.array);
}

class StringTypeParameterizedTestFixture
    : public ::testing::TestWithParam<std::tuple<ArrowDeviceType, enum ArrowType, bool>> {
 protected:
  std::pair<ArrowDeviceType, enum ArrowType> info;
};

std::tuple<ArrowDeviceType, enum ArrowType, bool> TestParams(ArrowDeviceType device_type,
                                                             enum ArrowType arrow_type,
                                                             bool include_null) {
  return {device_type, arrow_type, include_null};
}

TEST_P(StringTypeParameterizedTestFixture, ArrowDeviceCudaArrayViewString) {
  using namespace nanoarrow::literals;

  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceCuda(std::get<0>(GetParam()), 0);
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType string_type = std::get<1>(GetParam());
  bool include_null = std::get<2>(GetParam());
  int64_t expected_data_size;  // expected

  CudaStream stream(gpu->device_id);
  ASSERT_EQ(stream.Init(), NANOARROW_OK);

  // Create some test data
  ASSERT_EQ(ArrowArrayInitFromType(&array, string_type), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, "abc"_asv), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, "defg"_asv), NANOARROW_OK);
  if (include_null) {
    ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
    expected_data_size = 7;
  } else {
    ASSERT_EQ(ArrowArrayAppendString(&array, "hjk"_asv), NANOARROW_OK);
    expected_data_size = 10;
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array, nullptr), NANOARROW_OK);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, expected_data_size);
  EXPECT_EQ(device_array.array.length, 3);

  // Failing to provide a stream should error
  ASSERT_EQ(ArrowDeviceArrayViewCopyAsync(&device_array_view, gpu, nullptr, nullptr),
            EINVAL);

  // Copy required to Cuda
  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array, gpu, &device_array2), ENOTSUP);
  ASSERT_EQ(ArrowDeviceArrayViewCopyAsync(&device_array_view, gpu, &device_array2,
                                          stream.get()),
            NANOARROW_OK);
  ArrowArrayRelease(&device_array.array);

  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, gpu->device_id);
  ASSERT_EQ(
      ArrowDeviceArrayViewSetArrayMinimal(&device_array_view, &device_array2, nullptr),
      NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, -1);
  EXPECT_EQ(device_array_view.array_view.length, 3);
  EXPECT_EQ(device_array2.array.length, 3);

  // Copy required back to Cpu for Cuda; not for CudaHost
  if (gpu->device_type == ARROW_DEVICE_CUDA_HOST) {
    ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array2, cpu, &device_array),
              NANOARROW_OK);
  } else {
    ASSERT_EQ(ArrowDeviceArrayViewCopyAsync(&device_array_view, cpu, &device_array,
                                            stream.get()),
              NANOARROW_OK);
    ArrowArrayRelease(&device_array2.array);
  }

  ASSERT_NE(device_array.array.release, nullptr);
  ASSERT_EQ(device_array.device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  ASSERT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, expected_data_size);

  ASSERT_EQ(cuStreamSynchronize(*stream.get()), CUDA_SUCCESS);
  if (include_null) {
    EXPECT_EQ(
        memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefg", 7), 0);
  } else {
    EXPECT_EQ(
        memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefghjk", 7),
        0);
  }

  ArrowArrayRelease(&device_array.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowDeviceCuda, StringTypeParameterizedTestFixture,
    ::testing::Values(
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_STRING, true),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_STRING, false),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_STRING, true),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_STRING, false),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_BINARY, true),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_BINARY, false),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_BINARY, true),
        TestParams(ARROW_DEVICE_CUDA, NANOARROW_TYPE_LARGE_BINARY, false),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_STRING, true),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_STRING, false),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_LARGE_STRING, true),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_LARGE_STRING, false),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_BINARY, true),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_BINARY, false),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_LARGE_BINARY, true),
        TestParams(ARROW_DEVICE_CUDA_HOST, NANOARROW_TYPE_LARGE_BINARY, false)));
