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

from libc.stdint cimport uint8_t, int64_t
from libc.string cimport memcpy
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_IsValid
from cpython cimport (
    Py_buffer,
    PyObject_CheckBuffer,
    PyBuffer_Release,
)
from cpython.ref cimport Py_INCREF, Py_DECREF

from nanoarrow_c cimport *
from nanoarrow_device_c cimport *
from nanoarrow_dlpack cimport *

def c_version():
    """Return the nanoarrow C library version string
    """
    return ArrowNanoarrowVersion().decode("UTF-8")


# CPython utilities that are helpful in Python and not available in all
# implementations of ctypes (e.g., early Python versions, pypy)
def obj_is_capsule(obj, str name):
    return PyCapsule_IsValid(obj, name.encode()) == 1


def obj_is_buffer(obj):
    return PyObject_CheckBuffer(obj) == 1

class NanoarrowException(RuntimeError):
    """An error resulting from a call to the nanoarrow C library

    Calls to the nanoarrow C library and/or the Arrow C Stream interface
    callbacks return an errno error code and sometimes a message with extra
    detail. This exception wraps a RuntimeError to format a suitable message
    and store the components of the original error.
    """

    def __init__(self, what, code, message=""):
        self.what = what
        self.code = code
        self.message = message

        if self.message == "":
            super().__init__(f"{self.what} failed ({self.code})")
        else:
            super().__init__(f"{self.what} failed ({self.code}): {self.message}")

cdef class Error:
    """Memory holder for an ArrowError

    ArrowError is the C struct that is optionally passed to nanoarrow functions
    when a detailed error message might be returned. This class holds a C
    reference to the object and provides helpers for raising exceptions based
    on the contained message.
    """
    cdef ArrowError c_error

    def __cinit__(self):
        self.c_error.message[0] = 0

    def raise_message(self, what, code):
        """Raise a NanoarrowException from this message
        """
        raise NanoarrowException(what, code, self.c_error.message.decode("UTF-8"))

    def raise_message_not_ok(self, what, code):
        if code == NANOARROW_OK:
            return
        self.raise_message(what, code)

    @staticmethod
    def raise_error(what, code):
        """Raise a NanoarrowException without a message
        """
        raise NanoarrowException(what, code, "")

    @staticmethod
    def raise_error_not_ok(what, code):
        if code == NANOARROW_OK:
            return
        Error.raise_error(what, code)


# PyCapsule utilities
#
# PyCapsules are used (1) to safely manage memory associated with C structures
# by initializing them and ensuring the appropriate cleanup is invoked when
# the object is deleted; and (2) as an export mechanism conforming to the
# Arrow PyCapsule interface for the objects where this is defined.
cdef void pycapsule_schema_deleter(object schema_capsule) noexcept:
    cdef ArrowSchema* schema = <ArrowSchema*>PyCapsule_GetPointer(
        schema_capsule, 'arrow_schema'
    )
    if schema.release != NULL:
        ArrowSchemaRelease(schema)

    ArrowFree(schema)


cdef object alloc_c_schema(ArrowSchema** c_schema):
    c_schema[0] = <ArrowSchema*> ArrowMalloc(sizeof(ArrowSchema))
    # Ensure the capsule destructor doesn't call a random release pointer
    c_schema[0].release = NULL
    return PyCapsule_New(c_schema[0], 'arrow_schema', &pycapsule_schema_deleter)


cdef void pycapsule_array_deleter(object array_capsule) noexcept:
    cdef ArrowArray* array = <ArrowArray*>PyCapsule_GetPointer(
        array_capsule, 'arrow_array'
    )
    # Do not invoke the deleter on a used/moved capsule
    if array.release != NULL:
        ArrowArrayRelease(array)

    ArrowFree(array)


cdef object alloc_c_array(ArrowArray** c_array):
    c_array[0] = <ArrowArray*> ArrowMalloc(sizeof(ArrowArray))
    # Ensure the capsule destructor doesn't call a random release pointer
    c_array[0].release = NULL
    return PyCapsule_New(c_array[0], 'arrow_array', &pycapsule_array_deleter)


cdef void pycapsule_array_stream_deleter(object stream_capsule) noexcept:
    cdef ArrowArrayStream* stream = <ArrowArrayStream*>PyCapsule_GetPointer(
        stream_capsule, 'arrow_array_stream'
    )
    # Do not invoke the deleter on a used/moved capsule
    if stream.release != NULL:
        ArrowArrayStreamRelease(stream)

    ArrowFree(stream)


cdef object alloc_c_array_stream(ArrowArrayStream** c_stream):
    c_stream[0] = <ArrowArrayStream*> ArrowMalloc(sizeof(ArrowArrayStream))
    # Ensure the capsule destructor doesn't call a random release pointer
    c_stream[0].release = NULL
    return PyCapsule_New(c_stream[0], 'arrow_array_stream', &pycapsule_array_stream_deleter)


cdef void pycapsule_device_array_deleter(object device_array_capsule) noexcept:
    cdef ArrowDeviceArray* device_array = <ArrowDeviceArray*>PyCapsule_GetPointer(
        device_array_capsule, 'arrow_device_array'
    )
    # Do not invoke the deleter on a used/moved capsule
    if device_array.array.release != NULL:
        device_array.array.release(&device_array.array)

    ArrowFree(device_array)


cdef object alloc_c_device_array(ArrowDeviceArray** c_device_array):
    c_device_array[0] = <ArrowDeviceArray*> ArrowMalloc(sizeof(ArrowDeviceArray))
    # Ensure the capsule destructor doesn't call a random release pointer
    c_device_array[0].array.release = NULL
    return PyCapsule_New(c_device_array[0], 'arrow_device_array', &pycapsule_device_array_deleter)


cdef void pycapsule_array_view_deleter(object array_capsule) noexcept:
    cdef ArrowArrayView* array_view = <ArrowArrayView*>PyCapsule_GetPointer(
        array_capsule, 'nanoarrow_array_view'
    )

    ArrowArrayViewReset(array_view)

    ArrowFree(array_view)


cdef object alloc_c_array_view(ArrowArrayView** c_array_view):
    c_array_view[0] = <ArrowArrayView*> ArrowMalloc(sizeof(ArrowArrayView))
    ArrowArrayViewInitFromType(c_array_view[0], NANOARROW_TYPE_UNINITIALIZED)
    return PyCapsule_New(c_array_view[0], 'nanoarrow_array_view', &pycapsule_array_view_deleter)


# Provide a way to validate that we release all references we create
cdef int64_t pyobject_buffer_count = 0

def get_pyobject_buffer_count():
    global pyobject_buffer_count
    return pyobject_buffer_count


cdef void c_deallocate_pyobject_buffer(ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t size) noexcept with gil:
    Py_DECREF(<object>allocator.private_data)

    global pyobject_buffer_count
    pyobject_buffer_count -= 1


cdef void c_pyobject_buffer(object base, const void* buf, int64_t size_bytes, ArrowBuffer* out):
    out.data = <uint8_t*>buf
    out.size_bytes = size_bytes
    out.allocator = ArrowBufferDeallocator(
        <ArrowBufferDeallocatorCallback>c_deallocate_pyobject_buffer,
        <void*>base
    )
    Py_INCREF(base)

    global pyobject_buffer_count
    pyobject_buffer_count += 1


cdef void c_array_shallow_copy(object base, const ArrowArray* src, ArrowArray* dst):
    """Make the shallowest (safe) copy possible

    Once a CArray exists at the Python level, nanoarrow makes it very difficult
    to perform an operation that might render the pointed-to ArrowArray invalid.
    Performing a deep copy (i.e., copying buffer content) would be unexpected and
    prohibitively expensive, and performing a truly shallow copy (i.e., adding
    an ArrowArray implementation that simply PyINCREF/pyDECREFs the original array)
    is not safe because the Arrow C Data interface specification allows children
    to be "move"d. Even though nanoarrow's Python bindings do not do this unless
    explicitly requested, when passed to some other library they are free to do so.

    This implementation of a shallow copy creates a recursive copy of the original
    array, including any children and dictionary (if present). It uses the
    C library's ArrowArray implementation, which takes care of releasing children,
    and allows us to use the ArrowBufferDeallocator mechanism to add/remove
    references to the appropriate PyObject.
    """
    # Allocate an ArrowArray* that will definitely be cleaned up should an exception
    # be raised in the process of shallow copying its contents
    cdef ArrowArray* tmp
    shelter = alloc_c_array(&tmp)
    cdef int code

    code = ArrowArrayInitFromType(tmp, NANOARROW_TYPE_UNINITIALIZED)
    Error.raise_error_not_ok("ArrowArrayInitFromType()", code)

    # Copy data for this array, adding a reference for each buffer
    # This allows us to use the nanoarrow C library's ArrowArray
    # implementation without writing our own release callbacks/private_data.
    tmp.length = src.length
    tmp.offset = src.offset
    tmp.null_count = src.null_count

    for i in range(src.n_buffers):
        if src.buffers[i] != NULL:
            # The purpose of this buffer is soley so that we can use the
            # ArrowBufferDeallocator mechanism to add a reference to base.
            # The ArrowArray release callback that exists here after
            # because of ArrowArrayInitFromType() will call ArrowBufferReset()
            # on any buffer that was injected in this way (and thus release the
            # reference to base). We don't actually know the size of the buffer
            # (and our release callback doesn't use it), so it is set to 0.
            c_pyobject_buffer(base, src.buffers[i], 0, ArrowArrayBuffer(tmp, i))

        # The actual pointer value is tracked separately from the ArrowBuffer
        # (which is only concerned with object lifecycle).
        tmp.buffers[i] = src.buffers[i]

    tmp.n_buffers = src.n_buffers

    # Recursive shallow copy children
    if src.n_children > 0:
        code = ArrowArrayAllocateChildren(tmp, src.n_children)
        Error.raise_error_not_ok("ArrowArrayAllocateChildren()", code)

        for i in range(src.n_children):
            c_array_shallow_copy(base, src.children[i], tmp.children[i])

    # Recursive shallow copy dictionary
    if src.dictionary != NULL:
        code = ArrowArrayAllocateDictionary(tmp)
        Error.raise_error_not_ok("ArrowArrayAllocateDictionary()", code)

        c_array_shallow_copy(base, src.dictionary, tmp.dictionary)

    # Move tmp into dst
    ArrowArrayMove(tmp, dst)


cdef void c_device_array_shallow_copy(object base, const ArrowDeviceArray* src,
                                      ArrowDeviceArray* dst) noexcept:
    # Copy top-level information but leave the array marked as released
    # TODO: Should the sync event be copied here too?
    memcpy(dst, src, sizeof(ArrowDeviceArray))
    dst.array.release = NULL

    # Shallow copy the array
    c_array_shallow_copy(base, &src.array, &dst.array)


cdef void pycapsule_buffer_deleter(object stream_capsule) noexcept:
    cdef ArrowBuffer* buffer = <ArrowBuffer*>PyCapsule_GetPointer(
        stream_capsule, 'nanoarrow_buffer'
    )

    ArrowBufferReset(buffer)
    ArrowFree(buffer)


cdef object alloc_c_buffer(ArrowBuffer** c_buffer):
    c_buffer[0] = <ArrowBuffer*> ArrowMalloc(sizeof(ArrowBuffer))
    ArrowBufferInit(c_buffer[0])
    return PyCapsule_New(c_buffer[0], 'nanoarrow_buffer', &pycapsule_buffer_deleter)


cdef void c_deallocate_pybuffer(ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t size) noexcept with gil:
    cdef Py_buffer* buffer = <Py_buffer*>allocator.private_data
    PyBuffer_Release(buffer)
    ArrowFree(buffer)


cdef ArrowBufferAllocator c_pybuffer_deallocator(Py_buffer* buffer):
    # This should probably be changed in nanoarrow C; however, currently, the deallocator
    # won't get called if buffer.buf is NULL.
    if buffer.buf == NULL:
        PyBuffer_Release(buffer)
        return ArrowBufferAllocatorDefault()

    cdef Py_buffer* allocator_private = <Py_buffer*>ArrowMalloc(sizeof(Py_buffer))
    if allocator_private == NULL:
        PyBuffer_Release(buffer)
        raise MemoryError()

    memcpy(allocator_private, buffer, sizeof(Py_buffer))
    return ArrowBufferDeallocator(<ArrowBufferDeallocatorCallback>&c_deallocate_pybuffer, allocator_private)
