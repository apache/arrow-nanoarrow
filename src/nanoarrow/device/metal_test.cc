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

#include <Metal/Metal.hpp>

#include "nanoarrow/nanoarrow_device.hpp"

TEST(NanoarrowDeviceMetal, DefaultDevice) {
  nanoarrow::device::UniqueDevice device;
  ASSERT_EQ(ArrowDeviceMetalInitDefaultDevice(device.get(), nullptr), NANOARROW_OK);
  ASSERT_EQ(device->device_type, ARROW_DEVICE_METAL);
  ASSERT_NE(device->device_id, 0);

  ASSERT_EQ(ArrowDeviceMetalDefaultDevice(), ArrowDeviceMetalDefaultDevice());
}

TEST(NanoarrowDeviceMetal, DeviceGpuBufferInit) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView cpu_view = {data, sizeof(data)};

  struct ArrowBuffer buffer_aligned;
  ASSERT_EQ(ArrowDeviceMetalInitBuffer(&buffer_aligned), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppend(&buffer_aligned, data, sizeof(data)), NANOARROW_OK);
  struct ArrowBufferView gpu_view = {buffer_aligned.data, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

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

  ArrowBufferReset(&buffer_aligned);
}

TEST(NanoarrowDeviceMetal, DeviceGpuBufferMove) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
  struct ArrowBuffer buffer;
  struct ArrowBuffer buffer2;

  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView view = {data, sizeof(data)};

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, gpu, &buffer), NANOARROW_OK);

  // GPU -> GPU: just a move
  uint8_t* old_ptr = buffer.data;
  ASSERT_EQ(ArrowDeviceBufferMove(gpu, &buffer, gpu, &buffer2), NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(buffer2.data, old_ptr);
  EXPECT_EQ(buffer.data, nullptr);

  // GPU -> CPU: just a move
  ASSERT_EQ(ArrowDeviceBufferMove(gpu, &buffer2, cpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, 5);
  EXPECT_EQ(buffer.data, old_ptr);
  EXPECT_EQ(buffer2.data, nullptr);

  // CPU -> GPU: should be just a move here because the buffer is properly aligned
  // from the initial GPU allocation.
  ASSERT_EQ(ArrowDeviceBufferMove(cpu, &buffer, gpu, &buffer2), NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(buffer2.data, old_ptr);
  EXPECT_EQ(buffer.data, nullptr);
  ArrowBufferReset(&buffer2);

  // CPU -> GPU without alignment may require a copy
  ArrowBufferInit(&buffer);
  ASSERT_EQ(ArrowBufferAppend(&buffer, data, sizeof(data)), NANOARROW_OK);
  old_ptr = buffer.data;

  int code = ArrowDeviceBufferMove(cpu, &buffer, gpu, &buffer2);
  if (code == NANOARROW_OK) {
    // If the move was reported as a success, ensure it happened
    ASSERT_EQ(buffer2.data, old_ptr);
    EXPECT_EQ(buffer.data, nullptr);
    ASSERT_EQ(buffer2.size_bytes, 5);
    EXPECT_EQ(memcmp(buffer2.data, data, sizeof(data)), 0);
    ArrowBufferReset(&buffer2);
  } else {
    // Otherwise, ensure the old buffer was left intact
    ASSERT_EQ(buffer.data, old_ptr);
    EXPECT_EQ(buffer2.data, nullptr);
    ASSERT_EQ(buffer.size_bytes, 5);
    EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  }
}

TEST(NanoarrowDeviceMetal, DeviceGpuBufferCopy) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView cpu_view = {data, sizeof(data)};

  struct ArrowBuffer buffer;
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer), NANOARROW_OK);
  struct ArrowBufferView gpu_view = {buffer.data, sizeof(data)};

  struct ArrowBuffer buffer_dest;
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer_dest), NANOARROW_OK);
  struct ArrowBufferView gpu_dest_view = {buffer_dest.data, sizeof(data)};
  void* gpu_dest = buffer_dest.data;

  uint8_t cpu_dest[5];
  struct ArrowBufferView cpu_dest_view = {cpu_dest, sizeof(data)};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, gpu, gpu_dest_view), NANOARROW_OK);
  EXPECT_EQ(memcmp(gpu_dest, data, sizeof(data)), 0);
  memset(gpu_dest, 0, sizeof(data));

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, cpu, cpu_dest_view), NANOARROW_OK);
  EXPECT_EQ(memcmp(cpu_dest, data, sizeof(data)), 0);
  memset(cpu_dest, 0, sizeof(data));

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, cpu_view, gpu, gpu_dest_view), NANOARROW_OK);
  EXPECT_EQ(memcmp(gpu_dest, data, sizeof(data)), 0);

  ArrowBufferReset(&buffer);
  ArrowBufferReset(&buffer_dest);
}

TEST(NanoarrowDeviceMetal, DeviceAlignedBuffer) {
  struct ArrowBuffer buffer;
  int64_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  struct ArrowBufferView view = {data, sizeof(data)};

  ASSERT_EQ(ArrowDeviceMetalInitBuffer(&buffer), NANOARROW_OK);
  ASSERT_EQ(ArrowBufferAppendBufferView(&buffer, view), NANOARROW_OK);
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  EXPECT_EQ(buffer.capacity_bytes, 64);

  // Check that when we reallocate larger but less then the allocation size,
  // the pointer does not change
  uint8_t* old_ptr = buffer.data;
  ASSERT_EQ(ArrowBufferAppendBufferView(&buffer, view), NANOARROW_OK);
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  EXPECT_EQ(memcmp(buffer.data + sizeof(data), data, sizeof(data)), 0);
  EXPECT_EQ(buffer.capacity_bytes, 128);
  EXPECT_EQ(buffer.data, old_ptr);

  // But we can still shrink buffers with reallocation
  ASSERT_EQ(ArrowBufferResize(&buffer, 64, true), NANOARROW_OK);
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  EXPECT_NE(buffer.data, old_ptr);
  EXPECT_EQ(buffer.size_bytes, 64);
  EXPECT_EQ(buffer.capacity_bytes, 64);

  // When we reallocate to an invalid size, we get null
  ArrowBufferReset(&buffer);
  ASSERT_EQ(ArrowDeviceMetalInitBuffer(&buffer), NANOARROW_OK);
  EXPECT_EQ(ArrowBufferReserve(&buffer, std::numeric_limits<intptr_t>::max()), ENOMEM);
  EXPECT_EQ(buffer.data, nullptr);
  EXPECT_EQ(buffer.allocator.private_data, nullptr);
}

TEST(NanoarrowDeviceMetal, DeviceCpuArrayBuffers) {
  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAllocateChildren(array.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(array->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);

  ASSERT_EQ(ArrowDeviceMetalAlignArrayBuffers(array.get()), NANOARROW_OK);

  // Make sure we can build an array
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array->children[0], 1234), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  ASSERT_EQ(
      ArrowArrayFinishBuilding(array.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
      NANOARROW_OK);

  // Make sure that ArrowDeviceMetalInitArrayBuffers() copies existing content
  ASSERT_EQ(ArrowDeviceMetalAlignArrayBuffers(array.get()), NANOARROW_OK);

  auto data_ptr = reinterpret_cast<const int32_t*>(array->children[0]->buffers[1]);
  EXPECT_EQ(data_ptr[0], 1234);
}

class StringTypeParameterizedTestFixture
    : public ::testing::TestWithParam<enum ArrowType> {
 protected:
  enum ArrowType type;
};

TEST_P(StringTypeParameterizedTestFixture, ArrowDeviceMetalArrayViewString) {
  using namespace nanoarrow::literals;

  struct ArrowDevice* metal = ArrowDeviceMetalDefaultDevice();
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType string_type = GetParam();

  ASSERT_EQ(ArrowArrayInitFromType(&array, string_type), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, "abc"_asv), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, "defg"_asv), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array, nullptr), NANOARROW_OK);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);

  // In some MacOS environments, ArrowDeviceArrayMoveToDevice() with buffers allocated
  // using ArrowMalloc() will work (i.e., apparently MTL::Buffer can wrap
  // arbitrary bytes, but only sometimes)
  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;

  int code = ArrowDeviceArrayMoveToDevice(&device_array, metal, &device_array2);
  if (code == NANOARROW_OK) {
    // If the move was successful, ensure it actually happened
    ASSERT_EQ(device_array.array.release, nullptr);
    ASSERT_NE(device_array2.array.release, nullptr);
  } else {
    // If the move was not successful, ensure we can copy to a metal device array
    ASSERT_EQ(code, ENOTSUP);
    ASSERT_EQ(ArrowDeviceArrayViewCopy(&device_array_view, metal, &device_array2),
              NANOARROW_OK);
  }

  // Either way, ensure that the device array is reporting correct values
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array2, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);
  EXPECT_EQ(memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefg", 7),
            0);

  // Copy shouldn't be required back to the CPU
  ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array2, cpu, &device_array),
            NANOARROW_OK);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(memcmp(device_array_view.array_view.buffer_views[2].data.data, "abcdefg", 7),
            0);

  ArrowArrayRelease(&device_array.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowDeviceMetal, StringTypeParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_STRING,
                                           NANOARROW_TYPE_LARGE_STRING,
                                           NANOARROW_TYPE_BINARY,
                                           NANOARROW_TYPE_LARGE_BINARY));
