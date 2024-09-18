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

from nanoarrow._array import CDeviceArray
from nanoarrow._device import DEVICE_CPU, Device, DeviceType  # noqa: F401
from nanoarrow._schema import CSchemaBuilder
from nanoarrow.c_array import c_array, c_array_from_buffers
from nanoarrow.c_buffer import c_buffer
from nanoarrow.c_schema import c_schema


def cpu():
    return DEVICE_CPU


def resolve(device_type: DeviceType, device_id: int):
    return Device.resolve(DeviceType(device_type).value, device_id)


def c_device_array(obj, schema=None):
    """ArrowDeviceArray wrapper

    This class provides a user-facing interface to access the fields of an
    ArrowDeviceArray.

    These objects are created using :func:`c_device_array`, which accepts any
    device array or array-like object according to the Arrow device PyCapsule
    interface, the DLPack protocol, or any object accepted by :func:`c_array`.

    Parameters
    ----------
    obj : device array-like
        An object supporting the Arrow device PyCapsule interface, the DLPack
        protocol, or any object accepted by :func:`c_array`.
    schema : schema-like or None
        A schema-like object as sanitized by :func:`c_schema` or None.
    """
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

    if hasattr(obj, "__dlpack__"):
        buffer = c_buffer(obj, schema=schema)
        schema = CSchemaBuilder.allocate().set_type(buffer.data_type_id).finish()
        return c_array_from_buffers(
            schema,
            len(buffer),
            [None, buffer],
            null_count=0,
            move=True,
            device=buffer.device,
        )

    # Attempt to create a CPU array and wrap it
    cpu_array = c_array(obj, schema=schema)
    return CDeviceArray._init_from_array(cpu(), cpu_array._addr(), cpu_array.schema)
