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

from nanoarrow_c cimport *

cdef extern from "./dlpack_abi.h" nogil:
    cdef int DLPACK_VERSION

    ctypedef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLOpenCL
        kDLVulkan
        kDLMetal
        kDLVPI
        kDLROCM
        kDLROCMHost
        kDLExtDev
        kDLCUDAManaged
        kDLOneAPI
        kDLWebGPU
        kDLHexagon

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int32_t device_id

    cdef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor*)  # noqa: E211
