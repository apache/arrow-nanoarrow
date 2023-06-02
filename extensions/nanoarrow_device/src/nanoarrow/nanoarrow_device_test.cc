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
  EXPECT_EQ(cpu->device_id, 0);
  EXPECT_EQ(cpu, ArrowDeviceCpu());

  void* sync_event = nullptr;
  EXPECT_EQ(cpu->synchronize_event(cpu, cpu, sync_event, nullptr), NANOARROW_OK);
  sync_event = cpu;
  EXPECT_EQ(cpu->synchronize_event(cpu, cpu, sync_event, nullptr), EINVAL);
}

TEST(NanoarrowDevice, ArrowDeviceCpuBuffer) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowDeviceBufferView view = {data, 0, sizeof(data)};
  void* sync_event;

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, cpu, &buffer, &sync_event), NANOARROW_OK);
  EXPECT_EQ(buffer.size_bytes, 5);
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(memcmp(buffer.data, view.private_data, sizeof(data)), 0);

  struct ArrowBuffer buffer2;
  ASSERT_EQ(ArrowDeviceBufferMove(cpu, &buffer, cpu, &buffer2, &sync_event),
            NANOARROW_OK);
  EXPECT_EQ(buffer2.size_bytes, 5);
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(memcmp(buffer2.data, view.private_data, sizeof(data)), 0);
  EXPECT_EQ(buffer.data, nullptr);

  uint8_t dest[5];
  struct ArrowDeviceBufferView dest_view = {dest, 0, sizeof(dest)};
  ASSERT_EQ(ArrowDeviceBufferCopy(cpu, view, cpu, dest_view, &sync_event), NANOARROW_OK);
  EXPECT_EQ(sync_event, nullptr);
  EXPECT_EQ(memcmp(dest, view.private_data, sizeof(data)), 0);

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

  ArrowDeviceArrayInit(&device_array, cpu);
  ArrowArrayMove(&array, &device_array.array);

  ArrowDeviceArrayViewInit(&device_array_view);
  ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
  ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
            NANOARROW_OK);

  EXPECT_EQ(device_array_view.array_view.buffer_views[2].size_bytes, 7);

  // Copy shouldn't be required to the same device
  ASSERT_FALSE(ArrowDeviceArrayViewCopyRequired(&device_array_view, cpu));

  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array, &device_array_view, cpu, &device_array2),
      NANOARROW_OK);
  ASSERT_EQ(device_array.array.release, nullptr);
  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, cpu->device_id);

  device_array2.array.release(&device_array2.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowDevice, StringTypeParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_STRING,
                                           NANOARROW_TYPE_LARGE_STRING,
                                           NANOARROW_TYPE_BINARY,
                                           NANOARROW_TYPE_LARGE_BINARY));

class ListTypeParameterizedTestFixture : public ::testing::TestWithParam<enum ArrowType> {
 protected:
  enum ArrowType type;
};

TEST_P(ListTypeParameterizedTestFixture, ArrowDeviceCpuArrayViewList) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
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

  // Copy shouldn't be required to the same device
  ASSERT_FALSE(ArrowDeviceArrayViewCopyRequired(&device_array_view, cpu));

  struct ArrowDeviceArray device_array2;
  device_array2.array.release = nullptr;
  ASSERT_EQ(
      ArrowDeviceArrayTryMove(&device_array, &device_array_view, cpu, &device_array2),
      NANOARROW_OK);
  ASSERT_EQ(device_array.array.release, nullptr);
  ASSERT_NE(device_array2.array.release, nullptr);
  ASSERT_EQ(device_array2.device_id, cpu->device_id);

  schema.release(&schema);
  device_array2.array.release(&device_array2.array);
  ArrowDeviceArrayViewReset(&device_array_view);
}

INSTANTIATE_TEST_SUITE_P(NanoarrowDevice, ListTypeParameterizedTestFixture,
                         ::testing::Values(NANOARROW_TYPE_LIST,
                                           NANOARROW_TYPE_LARGE_LIST));

TEST(NanoarrowDevice, BasicStreamCpu) {
  struct ArrowSchema schema;
  struct ArrowArray array;
  struct ArrowArrayStream naive_stream;
  struct ArrowDeviceArray device_array;
  struct ArrowDeviceArrayStream array_stream;

  // Build schema, array, and naive_stream
  ASSERT_EQ(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(&array, 1234), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowBasicArrayStreamInit(&naive_stream, &schema, 1), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(&naive_stream, 0, &array);

  ASSERT_EQ(
      ArrowDeviceBasicArrayStreamInit(&array_stream, &naive_stream, ArrowDeviceCpu()),
      NANOARROW_OK);

  ASSERT_EQ(array_stream.get_schema(&array_stream, &schema), NANOARROW_OK);
  ASSERT_STREQ(schema.format, "i");
  schema.release(&schema);

  ASSERT_EQ(array_stream.get_next(&array_stream, &device_array), NANOARROW_OK);
  ASSERT_EQ(device_array.device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(device_array.device_id, 0);
  ASSERT_EQ(device_array.array.n_buffers, 2);
  device_array.array.release(&device_array.array);

  ASSERT_EQ(array_stream.get_last_error(&array_stream), nullptr);

  array_stream.release(&array_stream);
}
