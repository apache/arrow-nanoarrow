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

#include <cstdio>
#include <cstring>
#include <iostream>

// Usually, including either .h or .hpp is fine; however, use this version
// to make sure all the include paths used in all the files will work.
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_device.h"
#include "nanoarrow/nanoarrow_device.hpp"
#include "nanoarrow/nanoarrow_ipc.h"
#include "nanoarrow/nanoarrow_ipc.hpp"
#include "nanoarrow/nanoarrow_testing.hpp"

int main(int argc, char* argv[]) {
  // Use something from nanoarrow.hpp / libnanoarrow
  nanoarrow::UniqueSchema schema;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_INT32));
  std::printf("Schema format for int32 is '%s'\n", schema->format);

  // Use something from nanoarrow_device.hpp / libnanoarrow_device
  nanoarrow::device::UniqueDevice device;
  ArrowDeviceInitCpu(device.get());
  std::printf("Device ID of CPU device is %d\n", static_cast<int>(device->device_id));

  // Use something from nanoarrow_ipc.hpp / libnanoarrow_ipc
  nanoarrow::ipc::UniqueOutputStream output_stream;
  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamInitFile(output_stream.get(), stdout, 0));
  std::printf("Address of private data ptr is %p\n", output_stream->private_data);

  // Use something from nanoarrow_testing.hpp / libnanoarrow_testing
  nanoarrow::testing::TestingJSONWriter json_writer;
  NANOARROW_RETURN_NOT_OK(json_writer.WriteField(std::cout, schema.get()));
  std::cout << "\n";

  return EXIT_SUCCESS;
}
