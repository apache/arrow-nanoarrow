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
from libc.stdio cimport snprintf
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
    cdef object _obj
    cdef bint _close_obj
    cdef void* _addr
    cdef Py_ssize_t _size_bytes

    def __cinit__(self, obj, close_obj=False):
        self._obj = obj
        self._close_obj = close_obj
        self._addr = NULL
        self._size_bytes = 0

    @property
    def obj(self):
        return self._obj

    @property
    def close_obj(self):
        return self._close_obj

    def set_buffer(self, uintptr_t addr, Py_ssize_t size_bytes):
        self._addr = <void*>addr
        self._size_bytes = size_bytes

    # Needed for at least some implementations of readinto()
    def __len__(self):
        return self._size_bytes

    # Implement the buffer protocol so that this object can be used as
    # the argument to xxx.readinto(). This ensures that no extra copies
    # (beyond any buffering done by the upstream file-like object) are held
    # since the upstream object has access to the preallocated output buffer.
    # In this case, the preallocation is done by the ArrowArrayStream
    # implementation before issuing each read call (two per message, with
    # an extra call for a RecordBatch message to get the actual buffer data).
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        PyBuffer_FillInfo(buffer, self, self._addr, self._size_bytes, 0, flags)

    def __releasebuffer__(self, Py_buffer* buffer):
        pass


cdef ArrowErrorCode py_input_stream_read(ArrowIpcInputStream* stream, uint8_t* buf,
                                         int64_t buf_size_bytes, int64_t* size_read_out,
                                         ArrowError* error) noexcept nogil:

    with gil:
        stream_private = <object>stream.private_data
        stream_private.set_buffer(<uintptr_t>buf, buf_size_bytes)

        try:
            size_read_out[0] = stream_private.obj.readinto(stream_private)
            return NANOARROW_OK
        except Exception as e:
            cls = type(e).__name__.encode()
            msg = str(e).encode()
            snprintf(
                error.message,
                sizeof(error.message),
                "%s: %s",
                <const char*>cls,
                <const char*>msg
            )
            return EIO

cdef void py_input_stream_release(ArrowIpcInputStream* stream) noexcept nogil:
    with gil:
        stream_private = <object>stream.private_data
        if stream_private.close_obj:
            stream_private.obj.close()

        Py_DECREF(stream_private)

    stream.private_data = NULL
    stream.release = NULL


cdef class CIpcInputStream:
    cdef ArrowIpcInputStream _stream

    def __cinit__(self):
        self._stream.release = NULL

    def is_valid(self):
        return self._stream.release != NULL

    def __dealloc__(self):
        # Duplicating release() to avoid Python API calls in the deallocator
        if self._stream.release != NULL:
            self._stream.release(&self._stream)

    def release(self):
        if self._stream.release != NULL:
            self._stream.release(&self._stream)
            return True
        else:
            return False

    @staticmethod
    def from_readable(obj, close_obj=False):
        cdef CIpcInputStream stream = CIpcInputStream()
        cdef PyInputStreamPrivate private_data = PyInputStreamPrivate(obj, close_obj)

        stream._stream.private_data = <PyObject*>private_data
        Py_INCREF(private_data)
        stream._stream.read = &py_input_stream_read
        stream._stream.release = &py_input_stream_release
        return stream


def init_array_stream(CIpcInputStream input_stream, uintptr_t out):
    cdef ArrowArrayStream* out_ptr = <ArrowArrayStream*>out

    # There are some options here that could be exposed at some point
    cdef int code = ArrowIpcArrayStreamReaderInit(out_ptr, &input_stream._stream, NULL)
    if code != NANOARROW_OK:
        raise RuntimeError(f"ArrowIpcArrayStreamReaderInit() failed with code [{code}]")
