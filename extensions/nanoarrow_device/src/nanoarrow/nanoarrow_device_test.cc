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
  ASSERT_EQ(cpu->device_type, ARROW_DEVICE_CPU);
  ASSERT_EQ(cpu->device_id, 0);
  ASSERT_EQ(cpu, ArrowDeviceCpu());

  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView view = {data, sizeof(data)};
  void* sync_event;

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, cpu, &buffer, &sync_event), NANOARROW_OK);
  ASSERT_EQ(buffer.size_bytes, 5);
  ASSERT_EQ(sync_event, nullptr);
  ASSERT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  sync_event = &buffer;
  ASSERT_EQ(cpu->synchronize_event(cpu, cpu, sync_event, nullptr), EINVAL);
}

// Dummy device that can copy to/from the CPU
static ArrowErrorCode DummyNonCpuBufferInit(struct ArrowDevice* device_src,
                                            struct ArrowBufferView src,
                                            struct ArrowDevice* device_dst,
                                            struct ArrowBuffer* dst, void** sync_event) {
  if (device_src->device_type == ARROW_DEVICE_CPU &&
          device_dst->device_type == ARROW_DEVICE_EXT_DEV ||
      device_dst->device_type == ARROW_DEVICE_CPU &&
          device_src->device_type == ARROW_DEVICE_EXT_DEV) {
    ArrowBufferInit(dst);
    dst->allocator = ArrowBufferAllocatorDefault();
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppendBufferView(dst, src));
    *sync_event = NULL;
    return NANOARROW_OK;
  } else {
    return ENOTSUP;
  }
}

TEST(NanoarrowDevice, ArrowBufferCopyDummyNonCpuDevice) {
  struct ArrowDevice* cpu = ArrowDeviceCpu();
  struct ArrowDevice not_cpu;
  ArrowDeviceInitCpu(&not_cpu);
  not_cpu.device_type = ARROW_DEVICE_EXT_DEV;
  not_cpu.buffer_init = &DummyNonCpuBufferInit;

  struct ArrowBuffer buffer;
  uint8_t data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBufferView view = {data, sizeof(data)};
  void* sync_event;

  ASSERT_EQ(ArrowDeviceBufferInit(cpu, view, &not_cpu, &buffer, &sync_event),
            NANOARROW_OK);
  ASSERT_EQ(buffer.size_bytes, 5);
  ASSERT_EQ(sync_event, nullptr);
  ASSERT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  ASSERT_EQ(ArrowDeviceBufferInit(&not_cpu, view, cpu, &buffer, &sync_event),
            NANOARROW_OK);
  ASSERT_EQ(buffer.size_bytes, 5);
  ASSERT_EQ(sync_event, nullptr);
  ASSERT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  not_cpu.release(&not_cpu);
}

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
