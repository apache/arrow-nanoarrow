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

from libc.stdint cimport int32_t, int64_t

from nanoarrow_c cimport *

cdef extern from "nanoarrow_device.h" nogil:

    ctypedef int32_t ArrowDeviceType

    int32_t ARROW_DEVICE_CPU
    int32_t ARROW_DEVICE_CUDA
    int32_t ARROW_DEVICE_CUDA_HOST
    int32_t ARROW_DEVICE_OPENCL
    int32_t ARROW_DEVICE_VULKAN
    int32_t ARROW_DEVICE_METAL
    int32_t ARROW_DEVICE_VPI
    int32_t ARROW_DEVICE_ROCM
    int32_t ARROW_DEVICE_ROCM_HOST
    int32_t ARROW_DEVICE_EXT_DEV
    int32_t ARROW_DEVICE_CUDA_MANAGED
    int32_t ARROW_DEVICE_ONEAPI
    int32_t ARROW_DEVICE_WEBGPU
    int32_t ARROW_DEVICE_HEXAGON

    struct ArrowDeviceArray:
        ArrowArray array
        int64_t device_id
        ArrowDeviceType device_type
        void* sync_event

    struct ArrowDevice:
        ArrowDeviceType device_type
        int64_t device_id
        ArrowErrorCode (*array_init)(ArrowDevice* device,
                                     ArrowDeviceArray* device_array,
                                     ArrowArray* array)
        ArrowErrorCode (*array_move)(ArrowDevice* device_src,
                                     ArrowDeviceArray* src,
                                     ArrowDevice* device_dst,
                                     ArrowDeviceArray* dst)
        ArrowErrorCode (*buffer_init)(ArrowDevice* device_src,
                                      ArrowBufferView src,
                                      ArrowDevice* device_dst, ArrowBuffer* dst)
        ArrowErrorCode (*buffer_move)(ArrowDevice* device_src, ArrowBuffer* src,
                                      ArrowDevice* device_dst, ArrowBuffer* dst)
        ArrowErrorCode (*buffer_copy)(ArrowDevice* device_src,
                                      ArrowBufferView src,
                                      ArrowDevice* device_dst,
                                      ArrowBufferView dst)
        ArrowErrorCode (*synchronize_event)(ArrowDevice* device, void* sync_event,
                                            ArrowError* error)
        void (*release)(ArrowDevice* device)
        void* private_data


    struct ArrowDeviceArrayView:
        ArrowDevice* device
        ArrowArrayView array_view


    ArrowErrorCode ArrowDeviceArrayInit(ArrowDevice* device,
                                        ArrowDeviceArray* device_array,
                                        ArrowArray* array)

    void ArrowDeviceArrayViewInit(ArrowDeviceArrayView* device_array_view)

    ArrowErrorCode ArrowDeviceArrayViewSetArrayMinimal(ArrowDeviceArrayView* device_array_view,
                                                       ArrowDeviceArray* device_array,
                                                       ArrowError* error)

    ArrowErrorCode ArrowDeviceArrayViewSetArray(ArrowDeviceArrayView* device_array_view,
                                                ArrowDeviceArray* device_array,
                                                ArrowError* error)

    ArrowErrorCode ArrowDeviceArrayViewCopy(ArrowDeviceArrayView* src,
                                            ArrowDevice* device_dst,
                                            ArrowDeviceArray* dst)

    ArrowDevice* ArrowDeviceCpu()
