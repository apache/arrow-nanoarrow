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

TEST(NanoarrowDevice, CheckRuntime) {
  EXPECT_EQ(ArrowDeviceCheckRuntime(nullptr), NANOARROW_OK);
}

TEST(NanoarrowDevice, CpuDevice) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  EXPECT_EQ(cpu->device_type, ARROW_DEVICE_CPU);
  EXPECT_EQ(cpu->device_id, -1);
  EXPECT_EQ(cpu, ArrowDeviceCpu());

  void* sync_event = nullptr;
  EXPECT_EQ(cpu->synchronize_event(cpu, sync_event, nullptr), NANOARROW_OK);
  sync_event = cpu;
  EXPECT_EQ(cpu->synchronize_event(cpu, sync_event, nullptr), EINVAL);
}

TEST(NanoarrowDevice, ArrowDeviceCpuBuffer) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView view = {data, sizeof(data)};

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, cpu, &buffer), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, 5);
  EXPECT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);

  struct ArrowBuffer buffer2;
  ASSERT_EQ(ArrowDeviceBufferMove(cpu, &buffer, cpu, &buffer2), NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(memcmp(buffer2.data, view.data.data, sizeof(data)), 0);
  EXPECT_EQ(buffer.data, nullptr);

  uint8_t dest[5];
  struct ArrowBufferView dest_view = {dest, sizeof(dest)};
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, view, cpu, dest_view), NANOARROW_OK);
  EXPECT_EQ(memcmp(dest, view.data.data, sizeof(data)), 0);

  ArrowBufferReset(&buffer2);
}

class StringTypeParameterizedTestFixture
    : public ::testing::TestWithParam<enum ArrowType> {
 protected:
  enum ArrowType type;
};

TEST_P(StringTypeParameterizedTestFixture, ArrowDeviceCpuArrayViewString) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowArray array;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayView device_array_view;
  enum ArrowType string_type = GetParam();

  ASSERT_EQ(ArrowArrayInitFromType(&array, string_type), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("abc")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("defg")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

  ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array), NANOARROW_OK);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);

  // Copy shouldn't be required to the same device
  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(ArrowDeviceArrayMoveToDevice(&device_array, cpu, &device_array2),
            NANOARROW_OK);
  ASSERT_EQ(device_array.array.release, nullptr);
  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, cpu->device_id);

  ArrowArrayRelease(&device_array2.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowDevice, StringTypeParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_STRING,
                                           NANOARROW_TYPE_LARGE_STRING,
                                           NANOARROW_TYPE_BINARY,
                                           NANOARROW_TYPE_LARGE_BINARY));
