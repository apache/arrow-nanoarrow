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

#include "nanoarrow_device.hpp"

#include "nanoarrow_device_metal.h"

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
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};
  void* sync_event;

  auto mtl_device = reinterpret_cast<MTL::Device*>(gpu->private_data);
  MTL::Buffer* mtl_buffer_src =
      mtl_device->newBuffer(data, sizeof(data), MTL::ResourceStorageModeShared);
  struct ArrowDeviceBufferView gpu_view = {mtl_buffer_src, 0, sizeof(data)};

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(sync_event, nullptr);
  auto mtl_buffer = reinterpret_cast<MTL::Buffer*>(buffer.data);
  EXPECT_EQ(mtl_buffer->length(), sizeof(data));
  EXPECT_EQ(memcmp(mtl_buffer->contents(), data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, gpu, &buffer, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(sync_event, nullptr);
  mtl_buffer = reinterpret_cast<MTL::Buffer*>(buffer.data);
  EXPECT_EQ(mtl_buffer->length(), sizeof(data));
  EXPECT_EQ(memcmp(mtl_buffer->contents(), data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferInit(gpu, gpu_view, cpu, &buffer, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, sizeof(data));
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);
}

TEST(NanoarrowDeviceMetal, DeviceGpuBufferMove) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
  struct ArrowBuffer buffer;
  struct ArrowBuffer buffer2;

  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView view = {data, 0, sizeof(data)};
  void* sync_event;

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, gpu, &buffer, &sync_event), NANOARROW_OK);
  auto mtl_buffer = reinterpret_cast<MTL::Buffer*>(buffer.data);

  // GPU -> GPU: buffer data pointer stays the same
  ASSERT_EQ(ArrowDeviceBufferMove(gpu, &buffer, gpu, &buffer2, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(reinterpret_cast<MTL::Buffer*>(buffer2.data), mtl_buffer);
  EXPECT_EQ(buffer.data, nullptr);

  // GPU -> CPU: CPU buffer points to GPU buffer contents
  ASSERT_EQ(ArrowDeviceBufferMove(gpu, &buffer2, cpu, &buffer, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, 5);
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(reinterpret_cast<void*>(buffer.data), mtl_buffer->contents());
  EXPECT_EQ(buffer2.data, nullptr);

  // CPU -> GPU: introduces a copy; new buffer data is a new MTL::Buffer
  // with the same content but the old MTL::Buffer has been destroyed
  ASSERT_EQ(ArrowDeviceBufferMove(cpu, &buffer, gpu, &buffer2, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(sync_event, nullptr);
  auto mtl_buffer2 = reinterpret_cast<MTL::Buffer*>(buffer2.data);
  EXPECT_EQ(memcmp(mtl_buffer2->contents(), data, sizeof(data)), 0);
  EXPECT_EQ(buffer.data, nullptr);

  ArrowBufferReset(&buffer2);
}

TEST(NanoarrowDeviceMetal, DeviceGpuBufferCopy) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView cpu_view = {data, 0, sizeof(data)};
  void* sync_event;

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, cpu_view, gpu, &buffer, &sync_event),
            NANOARROW_OK);
  auto mtl_buffer = reinterpret_cast<MTL::Buffer*>(buffer.data);
  struct ArrowDeviceBufferView gpu_view = {mtl_buffer, 0, sizeof(data)};
  auto mtl_buffer_dest =
      mtl_buffer->device()->newBuffer(5, MTL::ResourceStorageModeShared);
  uint8_t* gpu_dest = reinterpret_cast<uint8_t*>(mtl_buffer_dest->contents());
  struct ArrowDeviceBufferView gpu_dest_view = {mtl_buffer_dest, 0, sizeof(data)};

  uint8_t cpu_dest[5];
  struct ArrowDeviceBufferView cpu_dest_view = {cpu_dest, 0, sizeof(data)};

  // GPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, gpu, gpu_dest_view, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(memcmp(gpu_dest, data, sizeof(data)), 0);
  memset(gpu_dest, 0, sizeof(data));

  // GPU -> CPU
  ASSERT_EQ(ArrowDeviceBufferCopy(gpu, gpu_view, cpu, cpu_dest_view, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(memcmp(cpu_dest, data, sizeof(data)), 0);
  memset(cpu_dest, 0, sizeof(data));

  // CPU -> GPU
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, cpu_view, gpu, gpu_dest_view, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(memcmp(gpu_dest, data, sizeof(data)), 0);

  mtl_buffer_dest->release();
  ArrowBufferReset(&buffer);
}

TEST(NanoarrowDeviceMetal, DeviceCpuBuffer) {
  struct ArrowBuffer buffer;
  int64_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  struct ArrowBufferView view = {nullptr, 0};

  ASSERT_EQ(ArrowDeviceMetalInitCpuBuffer(ArrowDeviceMetalDefaultDevice(), &buffer, view),
            NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, 0);
  EXPECT_EQ(buffer.capacity_bytes, 64);
  EXPECT_NE(buffer.data, nullptr);

  view = {data, sizeof(data)};
  ASSERT_EQ(ArrowBufferAppendBufferView(&buffer, view), NANOARROW_OK);
  EXPECT_EQ(memcmp(buffer.data, data, sizeof(data)), 0);
  EXPECT_EQ(buffer.capacity_bytes, 128);

  // Check that we can also reallocate smaller
  uint8_t* old_ptr = buffer.data;
  ASSERT_EQ(ArrowBufferResize(&buffer, 64, true), NANOARROW_OK);
  EXPECT_NE(buffer.data, old_ptr);
  EXPECT_EQ(buffer.size_bytes, 64);
  EXPECT_EQ(buffer.capacity_bytes, 64);

  // When we reallocate smaller than 64 bytes, the underlying buffer stays the same
  old_ptr = buffer.data;
  ASSERT_EQ(ArrowBufferResize(&buffer, 0, true), NANOARROW_OK);
  EXPECT_EQ(buffer.data, old_ptr);

  // When we reallocate to an invalid size, we get null
  EXPECT_EQ(ArrowBufferReserve(&buffer, std::numeric_limits<intptr_t>::max()), ENOMEM);
  EXPECT_EQ(buffer.data, nullptr);
  EXPECT_EQ(buffer.allocator.private_data, nullptr);

  EXPECT_EQ(ArrowDeviceMetalInitCpuBuffer(ArrowDeviceCpu(), &buffer, view), EINVAL);
}

TEST(NanoarrowDeviceMetal, DeviceCpuArrayBuffers) {
  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAllocateChildren(array.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(array->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);

  ASSERT_EQ(
      ArrowDeviceMetalInitCpuArrayBuffers(ArrowDeviceMetalDefaultDevice(), array.get()),
      NANOARROW_OK);

  // Make sure we can build an array
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(array->children[0], 1234), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  ASSERT_EQ(
      ArrowArrayFinishBuilding(array.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr),
      NANOARROW_OK);

  // Make sure that ArrowDeviceMetalInitArrayBuffers() copies existing content
  ASSERT_EQ(
      ArrowDeviceMetalInitCpuArrayBuffers(ArrowDeviceMetalDefaultDevice(), array.get()),
      NANOARROW_OK);

  auto data_ptr = reinterpret_cast<const int32_t*>(array->children[0]->buffers[1]);
  EXPECT_EQ(data_ptr[0], 1234);
}
