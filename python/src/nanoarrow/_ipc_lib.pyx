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
# cython: linetrace=True

from libc.stdint cimport uint8_t, int64_t, uintptr_t
from libc.errno cimport EIO
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython cimport Py_buffer, PyBuffer_FillInfo

from nanoarrow_c cimport (
    ArrowErrorCode,
    ArrowError,
    NANOARROW_OK,
    ArrowArrayStream,
)


cdef extern from "nanoarrow_ipc.h" nogil:
    struct ArrowIpcInputStream:
        ArrowErrorCode (*read)(ArrowIpcInputStream* stream, uint8_t* buf,
                               int64_t buf_size_bytes, int64_t* size_read_out,
                               ArrowError* error)
        void (*release)(ArrowIpcInputStream* stream)
        void* private_data

    struct ArrowIpcArrayStreamReaderOptions:
        int64_t field_index
        int use_shared_buffers

    ArrowErrorCode ArrowIpcArrayStreamReaderInit(
        ArrowArrayStream* out, ArrowIpcInputStream* input_stream,
        ArrowIpcArrayStreamReaderOptions* options)


cdef class PyInputStreamPrivate:
    cdef object obj
    cdef void* addr
    cdef Py_ssize_t size_bytes

    def __cinit__(self, obj):
        self.obj = obj
        self.addr = NULL
        self.size_bytes = 0

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        PyBuffer_FillInfo(buffer, self, self.addr, self.size_bytes, 0, flags)

    def __releasebuffer__(self, Py_buffer* buffer):
        pass



cdef ArrowErrorCode py_input_stream_read(ArrowIpcInputStream* stream, uint8_t* buf,
                                         int64_t buf_size_bytes, int64_t* size_read_out,
                                         ArrowError* error) noexcept:
    cdef PyInputStreamPrivate stream_private = <object>stream.private_data
    stream_private.addr = buf
    stream_private.size_bytes = buf_size_bytes

    try:
        size_read_out[0] = stream_private.obj.readinto(stream_private)
        return NANOARROW_OK
    except Exception as e:
        raise e
        return EIO


cdef void py_input_stream_release(ArrowIpcInputStream* stream) noexcept:
    cdef PyInputStreamPrivate stream_private = <object>stream.private_data
    Py_DECREF(stream_private)
    stream.private_data = NULL
    stream.release = NULL


cdef class CIpcInputStream:
    cdef ArrowIpcInputStream _stream

    def __cinit__(self):
        self._stream.release = NULL

    def __dealloc__(self):
        if self._stream.release != NULL:
            self._stream.release(&self._stream)


    def from_readable(obj):
        cdef CIpcInputStream stream = CIpcInputStream()

        cdef PyInputStreamPrivate private_data = PyInputStreamPrivate(obj)

        stream._stream.private_data = <PyObject*>private_data
        Py_INCREF(private_data)
        stream._stream.read = &py_input_stream_read
        stream._stream.release = &py_input_stream_release
        return stream


def init_array_stream(CIpcInputStream input_stream, uintptr_t out,
                      int field_index=-1, bint use_shared_buffers=True):
    cdef ArrowArrayStream* out_ptr = <ArrowArrayStream*>out

    cdef ArrowIpcArrayStreamReaderOptions options
    options.field_index = field_index
    options.use_shared_buffers = use_shared_buffers

    cdef int code = ArrowIpcArrayStreamReaderInit(out_ptr, &input_stream._stream, &options)
    if code != NANOARROW_OK:
        raise RuntimeError(f"ArrowIpcArrayStreamReaderInit() failed with code [{code}]")
