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
    ArrowBuffer,
    ArrowBufferView,
    ArrowType,
)

from nanoarrow._device cimport Device, CSharedSyncEvent


cdef class CBufferView:
    cdef object _base
    cdef ArrowBufferView _ptr
    cdef ArrowType _data_type
    cdef CSharedSyncEvent _event
    cdef Py_ssize_t _element_size_bits
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides
    cdef int64_t _n_elements
    cdef char _format[128]

    cdef _check_copy_into_bounds(self, Py_buffer* dest, int64_t offset, int64_t length,
                                 int64_t dest_offset, int64_t dest_itemsize)

    cdef Py_ssize_t _item_size(self)

    cdef _do_getbuffer(self, Py_buffer *buffer, int flags)

    cdef _do_releasebuffer(self, Py_buffer* buffer)

cdef class CBuffer:
    cdef object _base
    cdef ArrowBuffer* _ptr
    cdef ArrowType _data_type
    cdef int _element_size_bits
    cdef char _format[32]
    cdef Device _device
    cdef CBufferView _view
    cdef int _get_buffer_count

    cdef _assert_valid(self)

    cdef _assert_buffer_count_zero(self)

    cdef _populate_view(self)
