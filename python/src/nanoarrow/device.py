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

from nanoarrow._lib import CDEVICE_CPU, CDevice, CDeviceArray
from nanoarrow.c_lib import c_array


def cpu():
    return CDEVICE_CPU


def resolve(device_type, device_id):
    return CDevice.resolve(device_type, device_id)


def c_device_array(obj):
    if isinstance(obj, CDeviceArray):
        return obj

    # Only CPU for now
    cpu_array = c_array(obj)

    return cpu()._array_init(cpu_array._addr(), cpu_array.schema)
