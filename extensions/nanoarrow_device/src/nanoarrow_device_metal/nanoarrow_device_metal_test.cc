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

TEST(NanoarrowDeviceMetal, DeviceBuffer) {
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

TEST(NanoarrowDeviceMetal, DeviceArrayBuffers) {
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
