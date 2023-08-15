<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# nanoarrow device extension

This extension provides a similar set of tools as the core nanoarrow C API
extended to the
[Arrow C Device](https://arrow.apache.org/docs/dev/format/CDeviceDataInterface.html)
interfaces in the Arrow specification.

Currently, this extension provides an implementation of CUDA devices
and an implementation for the default Apple Metal device on MacOS/M1.
These implementation are preliminary/experimental and are under active
development.

## Example

```c
struct ArrowDevice* gpu = ArrowDeviceMetalDefaultDevice();
// Alternatively, ArrowDeviceCuda(ARROW_DEVICE_CUDA, 0)
// or  ArrowDeviceCuda(ARROW_DEVICE_CUDA_HOST, 0)
struct ArrowDevice* cpu = ArrowDeviceCpu();
struct ArrowArray array;
struct ArrowDeviceArray device_array;
struct ArrowDeviceArrayView device_array_view;

// Build a CPU array
ASSERT_EQ(ArrowArrayInitFromType(&array, NANOARROW_TYPE_STRING), NANOARROW_OK);
ASSERT_EQ(ArrowArrayStartAppending(&array), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("abc")), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendString(&array, ArrowCharView("defg")), NANOARROW_OK);
ASSERT_EQ(ArrowArrayAppendNull(&array, 1), NANOARROW_OK);
ASSERT_EQ(ArrowArrayFinishBuildingDefault(&array, nullptr), NANOARROW_OK);

// Convert to a DeviceArray, still on the CPU
ASSERT_EQ(ArrowDeviceArrayInit(cpu, &device_array, &array), NANOARROW_OK);

// Parse contents into a view that can be copied to another device
ArrowDeviceArrayViewInit(&device_array_view);
ArrowArrayViewInitFromType(&device_array_view.array_view, string_type);
ASSERT_EQ(ArrowDeviceArrayViewSetArray(&device_array_view, &device_array, nullptr),
          NANOARROW_OK);

// Copy to another device. For some devices, ArrowDeviceArrayMoveToDevice() is
// possible without an explicit copy (although this sometimes triggers an implicit
// copy by the driver).
struct ArrowDeviceArray device_array2;
device_array2.array.release = nullptr;
ASSERT_EQ(
    ArrowDeviceArrayViewCopy(&device_array, &device_array_view, gpu, &device_array2),
    NANOARROW_OK);
```
