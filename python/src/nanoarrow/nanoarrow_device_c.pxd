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


from libc.stdint cimport int32_t, int64_t
from nanoarrow_c cimport *

cdef extern from "nanoarrow/nanoarrow_device.h" nogil:

    ctypedef int32_t ArrowDeviceType

    cdef ArrowDeviceType ARROW_DEVICE_CPU
    cdef ArrowDeviceType ARROW_DEVICE_CUDA
    cdef ArrowDeviceType ARROW_DEVICE_CUDA_HOST
    cdef ArrowDeviceType ARROW_DEVICE_OPENCL
    cdef ArrowDeviceType ARROW_DEVICE_VULKAN
    cdef ArrowDeviceType ARROW_DEVICE_METAL
    cdef ArrowDeviceType ARROW_DEVICE_VPI
    cdef ArrowDeviceType ARROW_DEVICE_ROCM
    cdef ArrowDeviceType ARROW_DEVICE_ROCM_HOST
    cdef ArrowDeviceType ARROW_DEVICE_EXT_DEV
    cdef ArrowDeviceType ARROW_DEVICE_CUDA_MANAGED
    cdef ArrowDeviceType ARROW_DEVICE_ONEAPI
    cdef ArrowDeviceType ARROW_DEVICE_WEBGPU
    cdef ArrowDeviceType ARROW_DEVICE_HEXAGON

    struct ArrowDeviceArray:
        ArrowArray array
        int64_t device_id
        ArrowDeviceType device_type
        void* sync_event
        int64_t reserved[3]

    struct ArrowDevice:
        ArrowDeviceType device_type
        int64_t device_id
        ArrowErrorCode (*array_init)(ArrowDevice* device,
                               ArrowDeviceArray* device_array,
                               ArrowArray* array, void* sync_event, void* stream)
        ArrowErrorCode (*array_move)(ArrowDevice* device_src,
                               ArrowDeviceArray* src,
                               ArrowDevice* device_dst,
                               ArrowDeviceArray* dst)
        ArrowErrorCode (*buffer_init)(ArrowDevice* device_src,
                                ArrowBufferView src,
                                ArrowDevice* device_dst, ArrowBuffer* dst,
                                void* stream)
        ArrowErrorCode (*buffer_move)(ArrowDevice* device_src, ArrowBuffer* src,
                                ArrowDevice* device_dst, ArrowBuffer* dst)
        ArrowErrorCode (*buffer_copy)(ArrowDevice* device_src,
                                ArrowBufferView src,
                                ArrowDevice* device_dst,
                                ArrowBufferView dst, void* stream)
        ArrowErrorCode (*synchronize_event)(ArrowDevice* device, void* sync_event,
                                      void* stream, ArrowError* error)
        void (*release)(ArrowDevice* device)
        void* private_data


    ArrowDevice* ArrowDeviceCpu()
    ArrowDevice* ArrowDeviceResolve(ArrowDeviceType device_type, int64_t device_id)
    ArrowErrorCode ArrowDeviceArrayInit(ArrowDevice* device, ArrowDeviceArray* device_array, ArrowArray* array, void* sync_event)
