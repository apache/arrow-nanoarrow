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

from nanoarrow._lib import DEVICE_CPU, CDeviceArray, Device, DeviceType  # noqa: F401
from nanoarrow.c_lib import c_array, c_schema


def cpu():
    return DEVICE_CPU


def resolve(device_type, device_id):
    return Device.resolve(device_type, device_id)


def c_device_array(obj, schema=None):
    if schema is not None:
        schema = c_schema(schema)

    if isinstance(obj, CDeviceArray) and schema is None:
        return obj

    if hasattr(obj, "__arrow_c_device_array__"):
        schema_capsule = None if schema is None else schema.__arrow_c_schema__()
        schema_capsule, device_array_capsule = obj.__arrow_c_device_array__(
            requested_schema=schema_capsule
        )
        return CDeviceArray._import_from_c_capsule(schema_capsule, device_array_capsule)

    # Attempt to create a CPU array and wrap it
    cpu_array = c_array(obj, schema=schema)
    return cpu()._array_init(cpu_array._addr(), cpu_array.schema)
