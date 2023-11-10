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

import pyarrow as pa

import nanoarrow.device as na_device
import nanoarrow as na


def test_cpu_device():
    cpu = na._lib.Device.cpu()
    assert cpu.device_type == 1

    cpu = na._lib.Device.resolve(1, 0)
    assert cpu.device_type == 1

    pa_array = pa.array([1, 2, 3])

    darray = na_device.device_array(pa_array)
    assert darray.device_type == 1
    assert darray.device_id == 0
    assert darray.array.length == 3
