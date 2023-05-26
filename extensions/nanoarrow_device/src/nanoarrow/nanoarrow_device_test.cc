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

#include <stdio.h>

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

  ASSERT_EQ(cpu->copy_to(cpu, view, cpu, &buffer, &sync_event, nullptr), NANOARROW_OK);
  ASSERT_EQ(buffer.size_bytes, 5);
  ASSERT_EQ(sync_event, nullptr);
  ASSERT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  ASSERT_EQ(cpu->copy_from(cpu, &buffer, cpu, view, &sync_event, nullptr), NANOARROW_OK);
  ASSERT_EQ(buffer.size_bytes, 5);
  ASSERT_EQ(sync_event, nullptr);
  ASSERT_EQ(memcmp(buffer.data, view.data.data, sizeof(data)), 0);
  ArrowBufferReset(&buffer);

  sync_event = &buffer;
  ASSERT_EQ(cpu->synchronize_event(cpu, cpu, sync_event, nullptr), EINVAL);
}
