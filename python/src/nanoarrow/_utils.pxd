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

from libc.stdint cimport uint8_t, int64_t

from nanoarrow_c cimport (
    ArrowSchema,
    ArrowArray,
    ArrowArrayStream,
    ArrowArrayView,
    ArrowBuffer,
    ArrowError
)
from nanoarrow_device_c cimport ArrowDeviceArray

cdef object alloc_c_schema(ArrowSchema** c_schema)

cdef object alloc_c_array(ArrowArray** c_array)

cdef object alloc_c_array_stream(ArrowArrayStream** c_stream)

cdef object alloc_c_device_array(ArrowDeviceArray** c_device_array)

cdef object alloc_c_array_view(ArrowArrayView** c_array_view)

cdef object alloc_c_buffer(ArrowBuffer** c_buffer)

cdef void c_array_shallow_copy(object base, const ArrowArray* src, ArrowArray* dst)

cdef void c_device_array_shallow_copy(object base, const ArrowDeviceArray* src,
                                      ArrowDeviceArray* dst)

cdef object c_buffer_set_pybuffer(object obj, ArrowBuffer** c_buffer)

cdef void c_buffer_set_pyobject(object base, uint8_t* data, int64_t size_bytes, ArrowBuffer** c_buffer)

cdef class Error:
    cdef ArrowError c_error

    cdef raise_message(self, what, code)

    cdef raise_message_not_ok(self, what, code)

    @staticmethod
    cdef raise_error(what, code)

    @staticmethod
    cdef raise_error_not_ok(what, code)
