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

from libc.stdint cimport int64_t

from nanoarrow_c cimport (
    ArrowArray,
    ArrowArrayView,
)

from nanoarrow_device_c cimport (
    ArrowDeviceArray,
    ArrowDeviceType
)

from nanoarrow._device cimport CSharedSyncEvent
from nanoarrow._schema cimport CSchema


cdef class CArray:
    cdef object _base
    cdef ArrowArray* _ptr
    cdef CSchema _schema
    cdef ArrowDeviceType _device_type
    cdef int _device_id
    cdef void* _sync_event

    cdef _set_device(self, ArrowDeviceType device_type, int64_t device_id, void* sync_event)


cdef class CArrayView:
    cdef object _base
    cdef object _array_base
    cdef ArrowArrayView* _ptr
    cdef CSharedSyncEvent _event

cdef class CDeviceArray:
    cdef object _base
    cdef ArrowDeviceArray* _ptr
    cdef CSchema _schema
