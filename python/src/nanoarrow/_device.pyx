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

# cython: language_level = 3

from libc.stdint cimport uintptr_t, int64_t

from nanoarrow_device_c cimport (
    ARROW_DEVICE_CPU,
    ARROW_DEVICE_CUDA,
    ARROW_DEVICE_CUDA_HOST,
    ARROW_DEVICE_OPENCL,
    ARROW_DEVICE_VULKAN,
    ARROW_DEVICE_METAL,
    ARROW_DEVICE_VPI,
    ARROW_DEVICE_ROCM,
    ARROW_DEVICE_ROCM_HOST,
    ARROW_DEVICE_EXT_DEV,
    ARROW_DEVICE_CUDA_MANAGED,
    ARROW_DEVICE_ONEAPI,
    ARROW_DEVICE_WEBGPU,
    ARROW_DEVICE_HEXAGON,
    ArrowDevice,
    ArrowDeviceCpu,
    ArrowDeviceResolve
)

from nanoarrow._utils cimport Error

from enum import Enum

from nanoarrow import _repr_utils


class DeviceType(Enum):
    """
    An enumerator providing access to the device constant values
    defined in the Arrow C Device interface. Unlike the other enum
    accessors, this Python Enum is defined in Cython so that we can use
    the bulit-in functionality to do better printing of device identifiers
    for classes defined in Cython. Unlike the other enums, users don't
    typically need to specify these (but would probably like them printed
    nicely).
    """

    CPU = ARROW_DEVICE_CPU
    CUDA = ARROW_DEVICE_CUDA
    CUDA_HOST = ARROW_DEVICE_CUDA_HOST
    OPENCL = ARROW_DEVICE_OPENCL
    VULKAN =  ARROW_DEVICE_VULKAN
    METAL = ARROW_DEVICE_METAL
    VPI = ARROW_DEVICE_VPI
    ROCM = ARROW_DEVICE_ROCM
    ROCM_HOST = ARROW_DEVICE_ROCM_HOST
    EXT_DEV = ARROW_DEVICE_EXT_DEV
    CUDA_MANAGED = ARROW_DEVICE_CUDA_MANAGED
    ONEAPI = ARROW_DEVICE_ONEAPI
    WEBGPU = ARROW_DEVICE_WEBGPU
    HEXAGON = ARROW_DEVICE_HEXAGON


cdef class Device:
    """ArrowDevice wrapper

    The ArrowDevice structure is a nanoarrow internal struct (i.e.,
    not ABI stable) that contains callbacks for device operations
    beyond its type and identifier (e.g., copy buffers to or from
    a device).
    """

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowDevice*>addr

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Device) and
            other.device_type == self.device_type and
            other.device_id == self.device_id
        )

    def __repr__(self):
        return _repr_utils.device_repr(self)

    @property
    def device_type(self):
        return DeviceType(self._ptr.device_type)

    @property
    def device_type_id(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @staticmethod
    def resolve(device_type, int64_t device_id):
        if int(device_type) == ARROW_DEVICE_CPU:
            return DEVICE_CPU

        cdef ArrowDevice* c_device = ArrowDeviceResolve(device_type, device_id)
        if c_device == NULL:
            raise ValueError(f"Device not found for type {device_type}/{device_id}")

        return Device(None, <uintptr_t>c_device)


# Cache the CPU device
# The CPU device is statically allocated (so base is None)
DEVICE_CPU = Device(None, <uintptr_t>ArrowDeviceCpu())


cdef class CSharedSyncEvent:

    def __cinit__(self, Device device, uintptr_t sync_event=0):
        self.device = device
        self.sync_event = <void*>sync_event

    cdef synchronize(self):
        if self.sync_event == NULL:
            return

        cdef Error error = Error()
        cdef ArrowDevice* c_device = self.device._ptr
        cdef int code = c_device.synchronize_event(c_device, self.sync_event, NULL, &error.c_error)
        error.raise_message_not_ok("ArrowDevice::synchronize_event", code)

        self.sync_event = NULL

    cdef synchronize_stream(self, uintptr_t stream):
        cdef Error error = Error()
        cdef ArrowDevice* c_device = self.device._ptr
        cdef int code = c_device.synchronize_event(c_device, self.sync_event, <void*>stream, &error.c_error)
        error.raise_message_not_ok("ArrowDevice::synchronize_event with stream", code)
