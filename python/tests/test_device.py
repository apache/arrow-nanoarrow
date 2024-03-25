# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import nanoarrow as na
from nanoarrow import device


def test_cpu_device():
    cpu = device.cpu()
    assert cpu.device_type == 1
    assert cpu.device_id == 0
    assert "device_type: 1" in repr(cpu)

    cpu = device.resolve(1, 0)
    assert cpu.device_type == 1


def test_c_device_array():
    # Unrecognized arguments should be passed to c_array() to generate  CPU array
    darray = device.c_device_array([1, 2, 3], na.int32())

    assert darray.device_type == 1
    assert darray.device_id == 0
    assert darray.schema.format == "i"
    assert darray.array.length == 3
    assert darray.array.device_type == device.cpu().device_type
    assert darray.array.device_id == device.cpu().device_id
    assert "device_type: 1" in repr(darray)

    # A CDeviceArray should be returned as is
    assert device.c_device_array(darray) is darray

    # A CPU device array should be able to export to a regular array
    array = na.c_array(darray)
    assert array.schema.format == "i"
    assert array.buffers == darray.array.buffers


# Wrapper to prevent c_device_array() from returning early when it detects the
# input is already a CDeviceArray
class DeviceArrayWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __arrow_c_device_array__(self, requested_schema=None):
        return self.obj.__arrow_c_device_array__(requested_schema=requested_schema)


def test_c_device_array_protocol():
    darray = device.c_device_array([1, 2, 3], na.int32())
    wrapper = DeviceArrayWrapper(darray)

    darray2 = device.c_device_array(wrapper)
    assert darray2.schema.format == "i"
    assert darray2.array.length == 3
    assert darray2.array.buffers == darray.array.buffers
