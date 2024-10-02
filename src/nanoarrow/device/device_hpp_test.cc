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

#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow_device.hpp"

TEST(NanoarrowDeviceHpp, UniqueDeviceArray) {
  nanoarrow::device::UniqueDeviceArray array;
  ASSERT_EQ(array->array.release, nullptr);

  ASSERT_EQ(ArrowArrayInitFromType(&array->array, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_NE(array->array.release, nullptr);

  nanoarrow::device::UniqueDeviceArray array2 = std::move(array);
  ASSERT_EQ(array->array.release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  ASSERT_NE(array2->array.release, nullptr);
}

TEST(NanoarrowDeviceHpp, UniqueDeviceArrayStream) {
  nanoarrow::device::UniqueDeviceArrayStream stream;
  ASSERT_EQ(stream->release, nullptr);

  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_INT32), NANOARROW_OK);
  nanoarrow::UniqueArrayStream naive_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(naive_stream.get(), schema.get(), 0), NANOARROW_OK);

  ASSERT_EQ(
      ArrowDeviceBasicArrayStreamInit(stream.get(), naive_stream.get(), ArrowDeviceCpu()),
      NANOARROW_OK);
  ASSERT_NE(stream->release, nullptr);

  nanoarrow::device::UniqueDeviceArrayStream stream2 = std::move(stream);
  ASSERT_EQ(stream->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  ASSERT_NE(stream2->release, nullptr);
}

TEST(NanoarrowDeviceHpp, UniqueDevice) {
  nanoarrow::device::UniqueDevice device;
  ASSERT_EQ(device->release, nullptr);

  ArrowDeviceInitCpu(device.get());

  nanoarrow::device::UniqueDevice device2 = std::move(device);
  ASSERT_EQ(device->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
  ASSERT_NE(device2->release, nullptr);
}

TEST(NanoarrowDeviceHpp, UniqueDeviceArrayView) {
  nanoarrow::device::UniqueDeviceArrayView array_view;
  ASSERT_EQ(array_view->device, nullptr);
  ArrowDeviceArrayViewInit(array_view.get());
  ArrowArrayViewInitFromType(&array_view->array_view, NANOARROW_TYPE_INT32);

  ASSERT_EQ(array_view->array_view.storage_type, NANOARROW_TYPE_INT32);

  nanoarrow::device::UniqueDeviceArrayView array_view2 = std::move(array_view);
  ASSERT_EQ(array_view2->array_view.storage_type, NANOARROW_TYPE_INT32);
  ASSERT_EQ(array_view->array_view.storage_type,  // NOLINT(clang-analyzer-cplusplus.Move)
            NANOARROW_TYPE_UNINITIALIZED);
}
