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

"""Low-level nanoarrow Python bindings

This Cython extension provides low-level Python wrappers around the
Arrow C Data and Arrow C Stream interface structs. In general, there
is one wrapper per C struct and pointer validity is managed by keeping
strong references to Python objects. These wrappers are intended to
be literal and stay close to the structure definitions: higher level
interfaces can and should be built in Python where it is faster to
iterate and where it is easier to create a better user experience
by default (i.e., classes, methods, and functions implemented in Python
generally have better autocomplete + documentation available to IDEs).
"""

from libc.stdint cimport uintptr_t, uint8_t, int64_t
from libc.string cimport memcpy
from libc.stdio cimport snprintf
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString, PyBytes_Size
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_IsValid
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from cpython cimport (
    Py_buffer,
    PyObject_CheckBuffer,
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBuffer_ToContiguous,
    PyBuffer_FillInfo,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_FORMAT,
    PyBUF_WRITABLE,
    PyErr_WriteUnraisable,
    PyMem_Free,
    PyMem_Malloc
)
from cpython.ref cimport Py_INCREF, Py_DECREF
from nanoarrow_c cimport *
from nanoarrow_device_c cimport *
from nanoarrow_dlpack cimport *

from enum import Enum
from sys import byteorder as sys_byteorder
from struct import unpack_from, iter_unpack, calcsize, Struct
from nanoarrow import _repr_utils

def c_version():
    """Return the nanoarrow C library version string
    """
    return ArrowNanoarrowVersion().decode("UTF-8")


# CPython utilities that are helpful in Python and not available in all
# implementations of ctypes (e.g., early Python versions, pypy)
def _obj_is_capsule(obj, str name):
    return PyCapsule_IsValid(obj, name.encode()) == 1


def _obj_is_buffer(obj):
    return PyObject_CheckBuffer(obj) == 1


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


cdef void pycapsule_dlpack_deleter(object dltensor) noexcept:
    cdef DLManagedTensor* dlm_tensor

    # Do nothing if the capsule has been consumed
    if PyCapsule_IsValid(dltensor, "used_dltensor"):
        return

    dlm_tensor = <DLManagedTensor*>PyCapsule_GetPointer(dltensor, 'dltensor')
    if dlm_tensor == NULL:
        PyErr_WriteUnraisable(dltensor)
    # The deleter can be NULL if there is no way for the caller
    # to provide a reasonable destructor
    elif dlm_tensor.deleter:
        dlm_tensor.deleter(dlm_tensor)


cdef void view_dlpack_deleter(DLManagedTensor* tensor) noexcept with gil:
    if tensor.manager_ctx is NULL:
        return
    Py_DECREF(<CBufferView>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    PyMem_Free(tensor)


cdef DLDataType view_to_dlpack_data_type(CBufferView view):
    cdef DLDataType dtype
    # Define DLDataType struct
    if view.data_type in ('uint8', 'uint16', 'uint32', 'uint64'):
        dtype.code = kDLUInt
    elif view.data_type in ('int8', 'int16', 'int32', 'int64'):
        dtype.code = kDLInt
    elif view.data_type in ('half_float', 'float', 'double'):
        dtype.code = kDLFloat
    elif view.data_type == 'bool':
        raise ValueError('Bit-packed boolean data type not supported by DLPack.')
    else:
        raise ValueError('DataType is not compatible with DLPack spec: ' + view.data_type)
    dtype.lanes = <uint16_t>1
    dtype.bits = <uint8_t>(view.item_size * 8)

    return dtype


cdef object view_to_dlpack(CBufferView view):
    # Define DLDevice and DLDataType struct and
    # with that check for data type support first
    cdef DLDevice device = view_to_dlpack_device(view)
    cdef DLDataType dtype = view_to_dlpack_data_type(view)

    # Allocate memory for DLManagedTensor
    cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>PyMem_Malloc(sizeof(DLManagedTensor))
    # Define DLManagedTensor struct
    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*>view._ptr.data.data
    dl_tensor.ndim = 1

    dl_tensor.shape = &view._n_elements
    dl_tensor.strides = NULL
    dl_tensor.byte_offset = 0

    dl_tensor.device = device
    dl_tensor.dtype = dtype

    dlm_tensor.manager_ctx = <void*>view
    Py_INCREF(view)
    dlm_tensor.deleter = view_dlpack_deleter

    return PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_dlpack_deleter)


cdef DLDevice view_to_dlpack_device(CBufferView view):
    cdef DLDevice device

    # Check data type support
    if view.data_type == 'bool':
        raise ValueError('Bit-packed boolean data type not supported by DLPack.')
    elif view.data_type not in ('uint8', 'uint16', 'uint32', 'uint64',
                                'int8', 'int16', 'int32', 'int64',
                                'half_float', 'float', 'double'):
        raise ValueError('DataType is not compatible with DLPack spec: ' + view.data_type)

    # Define DLDevice struct
    if view._device.device_type_id == ARROW_DEVICE_CPU:
        device.device_type = kDLCPU
        device.device_id =  0
    else:
        raise ValueError('Only CPU device is currently supported.')

    return device


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


cdef c_arrow_type_from_format(format):
    # PyBuffer_SizeFromFormat() was added in Python 3.9 (potentially faster)
    item_size = calcsize(format)

    # Don't allow non-native endian values
    if sys_byteorder == "little" and (">" in format or "!" in format):
        raise ValueError(f"Can't convert format '{format}' to Arrow type")
    elif sys_byteorder == "big" and  "<" in format:
        raise ValueError(f"Can't convert format '{format}' to Arrow type")

    # Strip system endian specifiers
    format = format.strip("=@")

    if format == "c":
        return 0, NANOARROW_TYPE_STRING
    elif format == "e":
        return item_size, NANOARROW_TYPE_HALF_FLOAT
    elif format == "f":
        return item_size, NANOARROW_TYPE_FLOAT
    elif format == "d":
        return item_size, NANOARROW_TYPE_DOUBLE

    # Check for signed integers
    if format in ("b", "h", "i", "l", "q", "n"):
        if item_size == 1:
            return item_size, NANOARROW_TYPE_INT8
        elif item_size == 2:
            return item_size, NANOARROW_TYPE_INT16
        elif item_size == 4:
            return item_size, NANOARROW_TYPE_INT32
        elif item_size == 8:
            return item_size, NANOARROW_TYPE_INT64

    # Check for unsinged integers
    if format in ("B", "?", "H", "I", "L", "Q", "N"):
        if item_size == 1:
            return item_size, NANOARROW_TYPE_UINT8
        elif item_size == 2:
            return item_size, NANOARROW_TYPE_UINT16
        elif item_size == 4:
            return item_size, NANOARROW_TYPE_UINT32
        elif item_size == 8:
            return item_size, NANOARROW_TYPE_UINT64

    # If all else fails, return opaque fixed-size binary
    return item_size, NANOARROW_TYPE_BINARY


cdef int c_format_from_arrow_type(ArrowType type_id, int element_size_bits, size_t out_size, char* out):
    if type_id in (NANOARROW_TYPE_BINARY, NANOARROW_TYPE_FIXED_SIZE_BINARY) and element_size_bits > 0:
        snprintf(out, out_size, "%ds", <int>(element_size_bits // 8))
        return element_size_bits

    cdef const char* format_const = ""
    cdef int element_size_bits_calc = 0
    if type_id == NANOARROW_TYPE_STRING:
        format_const = "c"
        element_size_bits_calc = 0
    elif type_id == NANOARROW_TYPE_BINARY:
        format_const = "B"
        element_size_bits_calc = 0
    elif type_id == NANOARROW_TYPE_BOOL:
        # Bitmaps export as unspecified binary
        format_const = "B"
        element_size_bits_calc = 1
    elif type_id == NANOARROW_TYPE_INT8:
        format_const = "b"
        element_size_bits_calc = 8
    elif type_id == NANOARROW_TYPE_UINT8:
        format_const = "B"
        element_size_bits_calc = 8
    elif type_id == NANOARROW_TYPE_INT16:
        format_const = "h"
        element_size_bits_calc = 16
    elif type_id == NANOARROW_TYPE_UINT16:
        format_const = "H"
        element_size_bits_calc = 16
    elif type_id in (NANOARROW_TYPE_INT32, NANOARROW_TYPE_INTERVAL_MONTHS):
        format_const = "i"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_UINT32:
        format_const = "I"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_INT64:
        format_const = "q"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_UINT64:
        format_const = "Q"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_HALF_FLOAT:
        format_const = "e"
        element_size_bits_calc = 16
    elif type_id == NANOARROW_TYPE_FLOAT:
        format_const = "f"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_DOUBLE:
        format_const = "d"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_INTERVAL_DAY_TIME:
        format_const = "ii"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
        format_const = "iiq"
        element_size_bits_calc = 128
    elif type_id == NANOARROW_TYPE_DECIMAL128:
        format_const = "16s"
        element_size_bits_calc = 128
    elif type_id == NANOARROW_TYPE_DECIMAL256:
        format_const = "32s"
        element_size_bits_calc = 256
    else:
        raise ValueError(f"Unsupported Arrow type_id for format conversion: {type_id}")

    snprintf(out, out_size, "%s", format_const)
    return element_size_bits_calc


cdef object c_buffer_set_pybuffer(object obj, ArrowBuffer** c_buffer):
    ArrowBufferReset(c_buffer[0])

    cdef Py_buffer buffer
    cdef int rc = PyObject_GetBuffer(obj, &buffer, PyBUF_FORMAT | PyBUF_ANY_CONTIGUOUS)
    if rc != 0:
        raise BufferError()

    # Parse the buffer's format string to get the ArrowType and element size
    try:
        if buffer.format == NULL:
            format = "B"
        else:
            format = buffer.format.decode("UTF-8")
    except Exception as e:
        PyBuffer_Release(&buffer)
        raise e

    # Transfers ownership of buffer to c_buffer, whose finalizer will be called by
    # the capsule when the capsule is deleted or garbage collected
    c_buffer[0].data = <uint8_t*>buffer.buf
    c_buffer[0].size_bytes = <int64_t>buffer.len
    c_buffer[0].capacity_bytes = 0
    c_buffer[0].allocator = c_pybuffer_deallocator(&buffer)

    # Return the calculated components
    return format


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


# This could in theory use cpdef enum, but an initial attempt to do so
# resulted Cython duplicating some function definitions. For now, we resort
# to a more manual trampoline of values to make them accessible from
# schema.py.
cdef class CArrowType:
    """
    Wrapper around ArrowType to provide implementations in Python access
    to the values.
    """

    UNINITIALIZED = NANOARROW_TYPE_UNINITIALIZED
    NA = NANOARROW_TYPE_NA
    BOOL = NANOARROW_TYPE_BOOL
    UINT8 = NANOARROW_TYPE_UINT8
    INT8 = NANOARROW_TYPE_INT8
    UINT16 = NANOARROW_TYPE_UINT16
    INT16 = NANOARROW_TYPE_INT16
    UINT32 = NANOARROW_TYPE_UINT32
    INT32 = NANOARROW_TYPE_INT32
    UINT64 = NANOARROW_TYPE_UINT64
    INT64 = NANOARROW_TYPE_INT64
    HALF_FLOAT = NANOARROW_TYPE_HALF_FLOAT
    FLOAT = NANOARROW_TYPE_FLOAT
    DOUBLE = NANOARROW_TYPE_DOUBLE
    STRING = NANOARROW_TYPE_STRING
    BINARY = NANOARROW_TYPE_BINARY
    FIXED_SIZE_BINARY = NANOARROW_TYPE_FIXED_SIZE_BINARY
    DATE32 = NANOARROW_TYPE_DATE32
    DATE64 = NANOARROW_TYPE_DATE64
    TIMESTAMP = NANOARROW_TYPE_TIMESTAMP
    TIME32 = NANOARROW_TYPE_TIME32
    TIME64 = NANOARROW_TYPE_TIME64
    INTERVAL_MONTHS = NANOARROW_TYPE_INTERVAL_MONTHS
    INTERVAL_DAY_TIME = NANOARROW_TYPE_INTERVAL_DAY_TIME
    DECIMAL128 = NANOARROW_TYPE_DECIMAL128
    DECIMAL256 = NANOARROW_TYPE_DECIMAL256
    LIST = NANOARROW_TYPE_LIST
    STRUCT = NANOARROW_TYPE_STRUCT
    SPARSE_UNION = NANOARROW_TYPE_SPARSE_UNION
    DENSE_UNION = NANOARROW_TYPE_DENSE_UNION
    DICTIONARY = NANOARROW_TYPE_DICTIONARY
    MAP = NANOARROW_TYPE_MAP
    EXTENSION = NANOARROW_TYPE_EXTENSION
    FIXED_SIZE_LIST = NANOARROW_TYPE_FIXED_SIZE_LIST
    DURATION = NANOARROW_TYPE_DURATION
    LARGE_STRING = NANOARROW_TYPE_LARGE_STRING
    LARGE_BINARY = NANOARROW_TYPE_LARGE_BINARY
    LARGE_LIST = NANOARROW_TYPE_LARGE_LIST
    INTERVAL_MONTH_DAY_NANO = NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO


cdef class CArrowTimeUnit:
    """
    Wrapper around ArrowTimeUnit to provide implementations in Python access
    to the values.
    """

    SECOND = NANOARROW_TIME_UNIT_SECOND
    MILLI = NANOARROW_TIME_UNIT_MILLI
    MICRO = NANOARROW_TIME_UNIT_MICRO
    NANO = NANOARROW_TIME_UNIT_NANO


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

    cdef object _base
    cdef ArrowDevice* _ptr

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowDevice*>addr

    def _array_init(self, uintptr_t array_addr, CSchema schema):
        cdef ArrowArray* array_ptr = <ArrowArray*>array_addr
        cdef ArrowDeviceArray* device_array_ptr
        holder = alloc_c_device_array(&device_array_ptr)
        cdef int code = ArrowDeviceArrayInit(self._ptr, device_array_ptr, array_ptr)
        Error.raise_error_not_ok("ArrowDevice::init_array", code)

        return CDeviceArray(holder, <uintptr_t>device_array_ptr, schema)

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
        else:
            raise ValueError(f"Device not found for type {device_type}/{device_id}")


# Cache the CPU device
# The CPU device is statically allocated (so base is None)
DEVICE_CPU = Device(None, <uintptr_t>ArrowDeviceCpu())


cdef class CSchema:
    """Low-level ArrowSchema wrapper

    This object is a literal wrapper around a read-only ArrowSchema. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowSchema structures are always released).

    See `nanoarrow.c_schema()` for construction and usage examples.
    """
    # Currently, _base is always the capsule holding the root of a tree of ArrowSchemas
    # (but in general is just a strong reference to an object whose Python lifetime is
    # used to guarantee that _ptr is valid).
    cdef object _base
    cdef ArrowSchema* _ptr

    @staticmethod
    def allocate():
        cdef ArrowSchema* c_schema_out
        base = alloc_c_schema(&c_schema_out)
        return CSchema(base, <uintptr_t>(c_schema_out))

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowSchema*>addr

    def __deepcopy__(self):
        cdef CSchema out = CSchema.allocate()
        cdef int code = ArrowSchemaDeepCopy(self._ptr, out._ptr)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        return out

    @staticmethod
    def _import_from_c_capsule(schema_capsule):
        """
        Import from a ArrowSchema PyCapsule

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        """
        return CSchema(
            schema_capsule,
            <uintptr_t>PyCapsule_GetPointer(schema_capsule, 'arrow_schema')
        )

    def __arrow_c_schema__(self):
        """
        Export to a ArrowSchema PyCapsule
        """
        self._assert_valid()

        cdef:
            ArrowSchema* c_schema_out
            int result

        schema_capsule = alloc_c_schema(&c_schema_out)
        code = ArrowSchemaDeepCopy(self._ptr, c_schema_out)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy", code)
        return schema_capsule

    @property
    def _capsule(self):
        """
        Returns the capsule backing this CSchema or None if it does not exist
        or points to a parent ArrowSchema.
        """
        cdef ArrowSchema* maybe_capsule_ptr
        maybe_capsule_ptr = <ArrowSchema*>PyCapsule_GetPointer(self._base, 'arrow_schema')

        # This will return False if this is a child CSchema whose capsule holds
        # the parent ArrowSchema
        if maybe_capsule_ptr == self._ptr:
            return self._base

        return None


    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("schema is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("schema is released")

    def _to_string(self, int64_t max_chars=0, recursive=False):
        cdef int64_t n_chars
        if max_chars == 0:
            n_chars = ArrowSchemaToString(self._ptr, NULL, 0, recursive)
        else:
            n_chars = max_chars

        cdef char* out = <char*>ArrowMalloc(n_chars + 1)
        if not out:
            raise MemoryError()

        ArrowSchemaToString(self._ptr, out, n_chars + 1, recursive)
        out_str = out.decode("UTF-8")
        ArrowFree(out)

        return out_str

    def __repr__(self):
        return _repr_utils.schema_repr(self)

    def type_equals(self, CSchema other, check_nullability=False):
        self._assert_valid()

        if self._ptr == other._ptr:
            return True

        if self.format != other.format:
            return False

        # Nullability is not strictly part of the "type"; however, performing
        # this check recursively is verbose to otherwise accomplish and
        # sometimes this does matter.
        cdef int64_t flags = self.flags
        cdef int64_t other_flags = other.flags
        if not check_nullability:
            flags &= ~ARROW_FLAG_NULLABLE
            other_flags &= ~ARROW_FLAG_NULLABLE

        if flags != other_flags:
            return False

        if self.n_children != other.n_children:
            return False

        for child, other_child in zip(self.children, other.children):
            if not child.type_equals(other_child, check_nullability=check_nullability):
                return False

        if (self.dictionary is None) != (other.dictionary is None):
            return False

        if self.dictionary is not None:
            if not self.dictionary.type_equals(
                other.dictionary,
                check_nullability=check_nullability
            ):
                return False

        return True


    @property
    def format(self):
        self._assert_valid()
        if self._ptr.format != NULL:
            return self._ptr.format.decode("UTF-8")

    @property
    def name(self):
        self._assert_valid()
        if self._ptr.name != NULL:
            return self._ptr.name.decode("UTF-8")
        else:
            return None

    @property
    def flags(self):
        return self._ptr.flags

    @property
    def metadata(self):
        self._assert_valid()
        if self._ptr.metadata != NULL:
            return SchemaMetadata(self._base, <uintptr_t>self._ptr.metadata)
        else:
            return None

    @property
    def n_children(self):
        self._assert_valid()
        return self._ptr.n_children

    def child(self, int64_t i):
        self._assert_valid()
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")

        return CSchema(self._base, <uintptr_t>self._ptr.children[i])

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def dictionary(self):
        self._assert_valid()
        if self._ptr.dictionary != NULL:
            return CSchema(self, <uintptr_t>self._ptr.dictionary)
        else:
            return None

    def modify(self, *, format=None, name=None, flags=None, nullable=None,
               metadata=None, children=None, dictionary=None, validate=True):
        cdef CSchemaBuilder builder = CSchemaBuilder.allocate()

        if format is None:
            builder.set_format(self.format)
        else:
            builder.set_format(format)

        if name is None:
            builder.set_name(self.name)
        elif name is not False:
            builder.set_name(name)

        if flags is None:
            builder.set_flags(self.flags)
        else:
            builder.set_flags(flags)

        if nullable is not None:
            builder.set_nullable(nullable)

        if metadata is None:
            if self.metadata is not None:
                builder.append_metadata(self.metadata)
        else:
            builder.append_metadata(metadata)

        if children is None:
            if self.n_children > 0:
                builder.allocate_children(self.n_children)
                for i, child in enumerate(self.children):
                    builder.set_child(i, None, child)
        elif hasattr(children, "items"):
            builder.allocate_children(len(children))
            for i, item in enumerate(children.items()):
                name, child = item
                builder.set_child(i, name, child)
        else:
            builder.allocate_children(len(children))
            for i, child in enumerate(children):
                builder.set_child(i, None, child)

        if dictionary is None:
            if self.dictionary:
                builder.set_dictionary(self.dictionary)
        elif dictionary is not False:
            builder.set_dictionary(dictionary)

        if validate:
            builder.validate()

        return builder.finish()

# This is likely a better fit for a dedicated testing module; however, we need
# it in _lib.pyx to produce nice error messages when ensuring that one or
# more arrays conform to a given or inferred schema.
def assert_type_equal(actual, expected):
    if not isinstance(actual, CSchema):
        raise TypeError(f"expected is {type(actual).__name__}, not CSchema")

    if not isinstance(expected, CSchema):
        raise TypeError(f"expected is {type(expected).__name__}, not CSchema")

    if not actual.type_equals(expected):
        actual_label = actual._to_string(max_chars=80, recursive=True)
        expected_label = expected._to_string(max_chars=80, recursive=True)
        raise ValueError(
            f"Expected schema\n  '{expected_label}'"
            f"\nbut got\n  '{actual_label}'"
        )


cdef class CSchemaView:
    """Low-level ArrowSchemaView wrapper

    This object is a literal wrapper around a read-only ArrowSchemaView. It provides field accessors
    that return Python objects and handles structure lifecycle. Compared to an ArrowSchema,
    the nanoarrow ArrowSchemaView facilitates access to the deserialized content of an ArrowSchema
    (e.g., parameter values for parameterized types).

    See `nanoarrow.c_schema_view()` for construction and usage examples.
    """
    # _base is currently only a CSchema (but in general is just an object whose Python
    # lifetime guarantees that the pointed-to data from ArrowStringViews remains valid
    cdef object _base
    cdef ArrowSchemaView _schema_view
    # Not part of the ArrowSchemaView (but possibly should be)
    cdef bint _dictionary_ordered
    cdef bint _nullable
    cdef bint _map_keys_sorted

    _fixed_size_types = (
        NANOARROW_TYPE_FIXED_SIZE_LIST,
        NANOARROW_TYPE_FIXED_SIZE_BINARY
    )

    _decimal_types = (
        NANOARROW_TYPE_DECIMAL128,
        NANOARROW_TYPE_DECIMAL256
    )

    _time_unit_types = (
        NANOARROW_TYPE_TIME32,
        NANOARROW_TYPE_TIME64,
        NANOARROW_TYPE_DURATION,
        NANOARROW_TYPE_TIMESTAMP
    )

    _union_types = (
        NANOARROW_TYPE_DENSE_UNION,
        NANOARROW_TYPE_SPARSE_UNION
    )

    def __cinit__(self, CSchema schema):
        self._base = schema
        self._schema_view.type = NANOARROW_TYPE_UNINITIALIZED
        self._schema_view.storage_type = NANOARROW_TYPE_UNINITIALIZED

        cdef Error error = Error()
        cdef int code = ArrowSchemaViewInit(&self._schema_view, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowSchemaViewInit()", code)

        self._dictionary_ordered = schema._ptr.flags & ARROW_FLAG_DICTIONARY_ORDERED
        self._nullable = schema._ptr.flags & ARROW_FLAG_NULLABLE
        self._map_keys_sorted = schema._ptr.flags & ARROW_FLAG_MAP_KEYS_SORTED

    @property
    def layout(self):
        return CLayout(self, <uintptr_t>&self._schema_view.layout)

    @property
    def type_id(self):
        return self._schema_view.type

    @property
    def storage_type_id(self):
        return self._schema_view.storage_type

    @property
    def buffer_format(self):
        """The Python struct format representing an element of this type
        or None if there is no Python format string that can represent this
        type.
        """
        if self.extension_name or self._schema_view.type != self._schema_view.storage_type:
            return None

        cdef char out[128]
        cdef int element_size_bits = 0
        if self._schema_view.type == NANOARROW_TYPE_FIXED_SIZE_BINARY:
            element_size_bits = self._schema_view.fixed_size * 8

        try:
            c_format_from_arrow_type(self._schema_view.type, element_size_bits, sizeof(out), out)
            return out.decode()
        except ValueError:
            return None

    @property
    def type(self):
        cdef const char* type_str = ArrowTypeString(self._schema_view.type)
        if type_str != NULL:
            return type_str.decode()
        else:
            raise ValueError("ArrowTypeString() returned NULL")

    @property
    def storage_type(self):
        cdef const char* type_str = ArrowTypeString(self._schema_view.storage_type)
        if type_str != NULL:
            return type_str.decode()
        else:
            raise ValueError("ArrowTypeString() returned NULL")

    @property
    def dictionary_ordered(self):
        if self._schema_view.type == NANOARROW_TYPE_DICTIONARY:
            return self._dictionary_ordered != 0
        else:
            return None

    @property
    def nullable(self):
        return self._nullable != 0

    @property
    def map_keys_sorted(self):
        if self._schema_view.type == NANOARROW_TYPE_MAP:
            return self._map_keys_sorted != 0
        else:
            return None

    @property
    def fixed_size(self):
        if self._schema_view.type in CSchemaView._fixed_size_types:
            return self._schema_view.fixed_size
        else:
            return None

    @property
    def decimal_bitwidth(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_bitwidth
        else:
            return None

    @property
    def decimal_precision(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_precision
        else:
            return None

    @property
    def decimal_scale(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_scale
        else:
            return None

    @property
    def time_unit_id(self):
        if self._schema_view.type in CSchemaView._time_unit_types:
            return self._schema_view.time_unit
        else:
            return None

    @property
    def time_unit(self):
        if self._schema_view.type in CSchemaView._time_unit_types:
            return ArrowTimeUnitString(self._schema_view.time_unit).decode()
        else:
            return None

    @property
    def timezone(self):
        if self._schema_view.type == NANOARROW_TYPE_TIMESTAMP:
            return self._schema_view.timezone.decode()
        else:
            return None

    @property
    def union_type_ids(self):
        if self._schema_view.type in CSchemaView._union_types:
            type_ids_str = self._schema_view.union_type_ids.decode().split(',')
            return (int(type_id) for type_id in type_ids_str)
        else:
            return None

    @property
    def extension_name(self):
        if self._schema_view.extension_name.data != NULL:
            name_bytes = PyBytes_FromStringAndSize(
                self._schema_view.extension_name.data,
                self._schema_view.extension_name.size_bytes
            )
            return name_bytes.decode()
        else:
            return None

    @property
    def extension_metadata(self):
        if self._schema_view.extension_name.data != NULL:
            return PyBytes_FromStringAndSize(
                self._schema_view.extension_metadata.data,
                self._schema_view.extension_metadata.size_bytes
            )
        else:
            return None

    def __repr__(self):
        return _repr_utils.schema_view_repr(self)


cdef class CLayout:
    cdef ArrowLayout* _layout
    cdef object _base
    cdef int _n_buffers

    def __cinit__(self, base, uintptr_t ptr):
        self._base = base
        self._layout = <ArrowLayout*>ptr

        self._n_buffers = NANOARROW_MAX_FIXED_BUFFERS
        for i in range(NANOARROW_MAX_FIXED_BUFFERS):
            if self._layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE:
                self._n_buffers = i
                break

    @property
    def n_buffers(self):
        return self._n_buffers

    @property
    def buffer_data_type_id(self):
        return tuple(self._layout.buffer_data_type[i] for i in range(self._n_buffers))

    @property
    def element_size_bits(self):
        return tuple(self._layout.element_size_bits[i] for i in range(self._n_buffers))

    @property
    def child_size_elements(self):
        return self._layout.child_size_elements


cdef class CSchemaBuilder:
    cdef CSchema c_schema
    cdef ArrowSchema* _ptr

    def __cinit__(self, CSchema schema):
        self.c_schema = schema
        self._ptr = schema._ptr
        if self._ptr.release == NULL:
            ArrowSchemaInit(self._ptr)

    @staticmethod
    def allocate():
        return CSchemaBuilder(CSchema.allocate())

    def append_metadata(self, metadata):
        cdef CBuffer buffer = CBuffer.empty()

        cdef const char* existing_metadata = self.c_schema._ptr.metadata
        cdef int code = ArrowMetadataBuilderInit(buffer._ptr, existing_metadata)
        Error.raise_error_not_ok("ArrowMetadataBuilderInit()", code)

        cdef ArrowStringView key
        cdef ArrowStringView value
        cdef int32_t keys_added = 0

        for k, v in metadata.items():
            k = k.encode() if isinstance(k, str) else bytes(k)
            key.data = PyBytes_AsString(k)
            key.size_bytes = PyBytes_Size(k)

            v = v.encode() if isinstance(v, str) else bytes(v)
            value.data = PyBytes_AsString(v)
            value.size_bytes = PyBytes_Size(v)

            code = ArrowMetadataBuilderAppend(buffer._ptr, key, value)
            Error.raise_error_not_ok("ArrowMetadataBuilderAppend()", code)

            keys_added += 1

        if keys_added > 0:
            code = ArrowSchemaSetMetadata(self.c_schema._ptr, <const char*>buffer._ptr.data)
            Error.raise_error_not_ok("ArrowSchemaSetMetadata()", code)

        return self

    def child(self, int64_t i):
        return CSchemaBuilder(self.c_schema.child(i))

    def set_type(self, int type_id):
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetType(self._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowSchemaSetType()", code)

        return self

    def set_type_decimal(self, int type_id, int precision, int scale):
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetTypeDecimal(self._ptr, <ArrowType>type_id, precision, scale)
        Error.raise_error_not_ok("ArrowSchemaSetType()", code)

    def set_type_fixed_size(self, int type_id, int fixed_size):
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetTypeFixedSize(self._ptr, <ArrowType>type_id, fixed_size)
        Error.raise_error_not_ok("ArrowSchemaSetTypeFixedSize()", code)

        return self

    def set_type_date_time(self, int type_id, int time_unit, timezone):
        self.c_schema._assert_valid()

        cdef int code
        if timezone is None:
            code = ArrowSchemaSetTypeDateTime(self._ptr, <ArrowType>type_id, <ArrowTimeUnit>time_unit, NULL)
        else:
            timezone = str(timezone)
            code = ArrowSchemaSetTypeDateTime(self._ptr, <ArrowType>type_id, <ArrowTimeUnit>time_unit, timezone.encode("UTF-8"))

        Error.raise_error_not_ok("ArrowSchemaSetTypeDateTime()", code)

        return self

    def set_format(self, str format):
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetFormat(self._ptr, format.encode("UTF-8"))
        Error.raise_error_not_ok("ArrowSchemaSetFormat()", code)

        return self

    def set_name(self, name):
        self.c_schema._assert_valid()

        cdef int code
        if name is None:
            code = ArrowSchemaSetName(self._ptr, NULL)
        else:
            name = str(name)
            code = ArrowSchemaSetName(self._ptr, name.encode("UTF-8"))

        Error.raise_error_not_ok("ArrowSchemaSetName()", code)

        return self

    def allocate_children(self, int n):
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaAllocateChildren(self._ptr, n)
        Error.raise_error_not_ok("ArrowSchemaAllocateChildren()", code)

        return self

    def set_child(self, int64_t i, name, CSchema child_src):
        self.c_schema._assert_valid()

        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"Index out of range: {i}")

        if self._ptr.children[i].release != NULL:
            ArrowSchemaRelease(self._ptr.children[i])

        cdef int code = ArrowSchemaDeepCopy(child_src._ptr, self._ptr.children[i])
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        if name is not None:
            name = str(name)
            code = ArrowSchemaSetName(self._ptr.children[i], name.encode("UTF-8"))
            Error.raise_error_not_ok("ArrowSchemaSetName()", code)

        return self

    def set_dictionary(self, CSchema dictionary):
        self.c_schema._assert_valid()

        cdef int code
        if self._ptr.dictionary == NULL:
            code = ArrowSchemaAllocateDictionary(self._ptr)
            Error.raise_error_not_ok("ArrowSchemaAllocateDictionary()", code)

        if self._ptr.dictionary.release != NULL:
            ArrowSchemaRelease(self._ptr.dictionary)

        code = ArrowSchemaDeepCopy(dictionary._ptr, self._ptr.dictionary)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        return self

    def set_flags(self, flags):
        self._ptr.flags = flags
        return self

    def set_nullable(self, nullable):
        if nullable:
            self._ptr.flags = self._ptr.flags | ARROW_FLAG_NULLABLE
        else:
            self._ptr.flags = self._ptr.flags & ~ARROW_FLAG_NULLABLE

        return self

    def set_dictionary_ordered(self, dictionary_ordered):
        if dictionary_ordered:
            self._ptr.flags = self._ptr.flags | ARROW_FLAG_DICTIONARY_ORDERED
        else:
            self._ptr.flags = self._ptr.flags & ~ARROW_FLAG_DICTIONARY_ORDERED

        return self

    def validate(self):
        return CSchemaView(self.c_schema)

    def finish(self):
        self.c_schema._assert_valid()

        return self.c_schema


cdef class CArray:
    """Low-level ArrowArray wrapper

    This object is a literal wrapper around a read-only ArrowArray. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowArray structures are always released).

    See `nanoarrow.c_array()` for construction and usage examples.
    """
    cdef object _base
    cdef ArrowArray* _ptr
    cdef CSchema _schema
    cdef ArrowDeviceType _device_type
    cdef int _device_id

    @staticmethod
    def allocate(CSchema schema):
        cdef ArrowArray* c_array_out
        base = alloc_c_array(&c_array_out)
        return CArray(base, <uintptr_t>c_array_out, schema)

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowArray*>addr
        self._schema = schema
        self._device_type = ARROW_DEVICE_CPU
        self._device_id = 0

    cdef _set_device(self, ArrowDeviceType device_type, int64_t device_id):
        self._device_type = device_type
        self._device_id = device_id

    @staticmethod
    def _import_from_c_capsule(schema_capsule, array_capsule):
        """
        Import from a ArrowSchema and ArrowArray PyCapsule tuple.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        array_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_array' containing an
            ArrowArray pointer.
        """
        cdef:
            CSchema out_schema
            CArray out

        out_schema = CSchema._import_from_c_capsule(schema_capsule)
        out = CArray(
            array_capsule,
            <uintptr_t>PyCapsule_GetPointer(array_capsule, 'arrow_array'),
            out_schema
        )

        return out

    def __getitem__(self, k):
        self._assert_valid()

        if not isinstance(k, slice):
            raise TypeError(
                f"Can't subset CArray with object of type {type(k).__name__}")

        if k.step is not None:
            raise ValueError("Can't slice CArray with step")

        cdef int64_t start = 0 if k.start is None else k.start
        cdef int64_t stop = self._ptr.length if k.stop is None else k.stop
        if start < 0:
            start = self._ptr.length + start
        if stop < 0:
            stop = self._ptr.length + stop

        if start > self._ptr.length or stop > self._ptr.length or stop < start:
            raise IndexError(
                f"{k} does not describe a valid slice of CArray "
                f"with length {self._ptr.length}"
            )

        cdef ArrowArray* c_array_out
        base = alloc_c_array(&c_array_out)
        c_array_shallow_copy(self._base, self._ptr, c_array_out)

        c_array_out.offset = c_array_out.offset + start
        c_array_out.length = stop - start
        cdef CArray out = CArray(base, <uintptr_t>c_array_out, self._schema)
        out._set_device(self._device_type, self._device_id)
        return out

    def __arrow_c_array__(self, requested_schema=None):
        """
        Get a pair of PyCapsules containing a C ArrowArray representation of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowArray,
            respectively.
        """
        self._assert_valid()

        if self._device_type != ARROW_DEVICE_CPU:
            raise ValueError(
                "Can't invoke __arrow_c_array__ on non-CPU array "
                f"with device_type {self._device_type}")

        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        # Export a shallow copy pointing to the same data in a way
        # that ensures this object stays valid.

        # TODO optimize this to export a version where children are reference
        # counted and can be released separately
        cdef ArrowArray* c_array_out
        array_capsule = alloc_c_array(&c_array_out)
        c_array_shallow_copy(self._base, self._ptr, c_array_out)

        return self._schema.__arrow_c_schema__(), array_capsule

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("CArray is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("CArray is released")

    def view(self):
        device = Device.resolve(self._device_type, self._device_id)
        return CArrayView.from_array(self, device)

    @property
    def schema(self):
        return self._schema

    @property
    def device_type(self):
        return DeviceType(self._device_type)

    @property
    def device_type_id(self):
        return self._device_type

    @property
    def device_id(self):
        return self._device_id

    def __len__(self):
        self._assert_valid()
        return self._ptr.length

    @property
    def length(self):
        return len(self)

    @property
    def offset(self):
        self._assert_valid()
        return self._ptr.offset

    @property
    def null_count(self):
        self._assert_valid()
        return self._ptr.null_count

    @property
    def n_buffers(self):
        self._assert_valid()
        return self._ptr.n_buffers

    @property
    def buffers(self):
        self._assert_valid()
        return tuple(<uintptr_t>self._ptr.buffers[i] for i in range(self._ptr.n_buffers))

    @property
    def n_children(self):
        self._assert_valid()
        return self._ptr.n_children

    def child(self, int64_t i):
        self._assert_valid()
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")
        cdef CArray out = CArray(
            self._base,
            <uintptr_t>self._ptr.children[i],
            self._schema.child(i)
        )
        out._set_device(self._device_type, self._device_id)
        return out

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def dictionary(self):
        self._assert_valid()
        cdef CArray out
        if self._ptr.dictionary != NULL:
            out = CArray(self, <uintptr_t>self._ptr.dictionary, self._schema.dictionary)
            out._set_device(self._device_type, self._device_id)
            return out
        else:
            return None

    def __repr__(self):
        return _repr_utils.array_repr(self)


cdef class CArrayView:
    """Low-level ArrowArrayView wrapper

    This object is a literal wrapper around an ArrowArrayView. It provides field accessors
    that return Python objects and handles the structure lifecycle (i.e., initialized
    ArrowArrayView structures are always released).

    See `nanoarrow.c_array_view()` for construction and usage examples.
    """
    cdef object _base
    cdef object _array_base
    cdef ArrowArrayView* _ptr
    cdef Device _device

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayView*>addr
        self._device = DEVICE_CPU

    def _set_array(self, CArray array, Device device=DEVICE_CPU):
        cdef Error error = Error()
        cdef int code

        if device is DEVICE_CPU:
            code = ArrowArrayViewSetArray(self._ptr, array._ptr, &error.c_error)
        else:
            code = ArrowArrayViewSetArrayMinimal(self._ptr, array._ptr, &error.c_error)

        error.raise_message_not_ok("ArrowArrayViewSetArray()", code)
        self._array_base = array._base
        self._device = device
        return self

    @property
    def storage_type_id(self):
        return self._ptr.storage_type

    @property
    def storage_type(self):
        cdef const char* type_str = ArrowTypeString(self._ptr.storage_type)
        if type_str != NULL:
            return type_str.decode('UTF-8')

    @property
    def layout(self):
        return CLayout(self, <uintptr_t>&self._ptr.layout)

    def __len__(self):
        return self._ptr.length

    @property
    def length(self):
        return len(self)

    @property
    def offset(self):
        return self._ptr.offset

    @property
    def null_count(self):
        return self._ptr.null_count

    @property
    def n_children(self):
        return self._ptr.n_children

    def child(self, int64_t i):
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")

        cdef CArrayView child = CArrayView(
            self._base,
            <uintptr_t>self._ptr.children[i]
        )

        child._device = self._device
        return child

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def n_buffers(self):
        return self.layout.n_buffers

    def buffer_type(self, int64_t i):
        if i < 0 or i >= self.n_buffers:
            raise IndexError(f"{i} out of range [0, {self.n_buffers}]")

        buffer_type = self._ptr.layout.buffer_type[i]
        if buffer_type == NANOARROW_BUFFER_TYPE_VALIDITY:
            return "validity"
        elif buffer_type == NANOARROW_BUFFER_TYPE_TYPE_ID:
            return "type_id"
        elif buffer_type == NANOARROW_BUFFER_TYPE_UNION_OFFSET:
            return "union_offset"
        elif buffer_type == NANOARROW_BUFFER_TYPE_DATA_OFFSET:
            return "data_offset"
        elif buffer_type == NANOARROW_BUFFER_TYPE_DATA:
            return "data"
        else:
            return "none"

    def buffer(self, int64_t i):
        if i < 0 or i >= self.n_buffers:
            raise IndexError(f"{i} out of range [0, {self.n_buffers}]")

        cdef ArrowBufferView* buffer_view = &(self._ptr.buffer_views[i])

        # Check the buffer size here because the error later is cryptic.
        # Buffer sizes are set to -1 when they are "unknown", so because of errors
        # in nanoarrow/C or because the array is on a non-CPU device, that -1 value
        # could leak its way here.
        if buffer_view.size_bytes < 0:
            raise RuntimeError(f"ArrowArrayView buffer {i} has size_bytes < 0")

        return CBufferView(
            self._array_base,
            <uintptr_t>buffer_view.data.data,
            buffer_view.size_bytes,
            self._ptr.layout.buffer_data_type[i],
            self._ptr.layout.element_size_bits[i],
            self._device
        )

    @property
    def buffers(self):
        for i in range(self.n_buffers):
            yield self.buffer(i)

    @property
    def dictionary(self):
        if self._ptr.dictionary == NULL:
            return None
        else:
            return CArrayView(
                self,
                <uintptr_t>self._ptr.dictionary
            )

    def __repr__(self):
        return _repr_utils.array_view_repr(self)

    @staticmethod
    def from_schema(CSchema schema):
        cdef ArrowArrayView* c_array_view
        base = alloc_c_array_view(&c_array_view)

        cdef Error error = Error()
        cdef int code = ArrowArrayViewInitFromSchema(c_array_view,
                                                     schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayViewInitFromSchema()", code)

        return CArrayView(base, <uintptr_t>c_array_view)

    @staticmethod
    def from_array(CArray array, Device device=DEVICE_CPU):
        out = CArrayView.from_schema(array._schema)
        return out._set_array(array, device)


cdef class SchemaMetadata:
    """Wrapper for a lazily-parsed CSchema.metadata string
    """

    cdef object _base
    cdef const char* _metadata
    cdef ArrowMetadataReader _reader

    def __cinit__(self, object base, uintptr_t ptr):
        self._base = base
        self._metadata = <const char*>ptr

    @staticmethod
    def empty():
        return SchemaMetadata(None, 0)

    cdef _init_reader(self):
        cdef int code = ArrowMetadataReaderInit(&self._reader, self._metadata)
        Error.raise_error_not_ok("ArrowMetadataReaderInit()", code)

    def __len__(self):
        self._init_reader()
        return self._reader.remaining_keys

    def __contains__(self, item):
        for key, _ in self.items():
            if item == key:
                return True

        return False

    def __getitem__(self, k):
        out = None

        for key, value in self.items():
            if k == key:
                if out is None:
                    out = value
                else:
                    raise KeyError(f"key {k} matches more than one value in metadata")

        return out

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def keys(self):
        return list(self)

    def values(self):
        return [value for _, value in self.items()]

    def items(self):
        cdef ArrowStringView key
        cdef ArrowStringView value
        self._init_reader()
        while self._reader.remaining_keys > 0:
            ArrowMetadataReaderRead(&self._reader, &key, &value)
            key_obj = PyBytes_FromStringAndSize(key.data, key.size_bytes)
            value_obj = PyBytes_FromStringAndSize(value.data, value.size_bytes)
            yield key_obj, value_obj

    def __repr__(self):
        lines = [
            f"<{_repr_utils.make_class_label(self)}>",
            _repr_utils.metadata_repr(self)
        ]
        return "\n".join(lines)


cdef class CBufferView:
    """Wrapper for Array buffer content

    This object is a Python wrapper around a buffer held by an Array.
    It implements the Python buffer protocol and is best accessed through
    another implementor (e.g., `np.array(array_view.buffers[1])`)). Note that
    this buffer content does not apply any parent offset.
    """
    cdef object _base
    cdef ArrowBufferView _ptr
    cdef ArrowType _data_type
    cdef Device _device
    cdef Py_ssize_t _element_size_bits
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides
    cdef int64_t _n_elements
    cdef char _format[128]

    def __cinit__(self, object base, uintptr_t addr, int64_t size_bytes,
                  ArrowType data_type,
                  Py_ssize_t element_size_bits, Device device):
        self._base = base
        self._ptr.data.data = <void*>addr
        self._ptr.size_bytes = size_bytes
        self._data_type = data_type
        self._device = device
        self._format[0] = 0
        self._element_size_bits = c_format_from_arrow_type(
            self._data_type,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._strides = self._item_size()
        self._shape = self._ptr.size_bytes // self._strides
        if self._data_type == NANOARROW_TYPE_BOOL:
            self._n_elements = self._shape * 8
        else:
            self._n_elements = self._shape

    def _addr(self):
        return <uintptr_t>self._ptr.data.data

    @property
    def device(self):
        return self._device

    @property
    def element_size_bits(self):
        return self._element_size_bits

    @property
    def size_bytes(self):
        return self._ptr.size_bytes

    @property
    def data_type_id(self):
        return self._data_type

    @property
    def data_type(self):
        return ArrowTypeString(self._data_type).decode("UTF-8")

    @property
    def format(self):
        return self._format.decode("UTF-8")

    @property
    def item_size(self):
        return self._strides

    def __len__(self):
        return self._shape

    def __getitem__(self, int64_t i):
        if i < 0 or i >= self._shape:
            raise IndexError(f"Index {i} out of range")
        cdef int64_t offset = self._strides * i
        value = unpack_from(self.format, buffer=self, offset=offset)
        if len(value) == 1:
            return value[0]
        else:
            return value

    def __iter__(self):
        return self._iter_dispatch(0, len(self))

    def _iter_dispatch(self, int64_t offset, int64_t length):
        if offset < 0 or length < 0 or (offset + length) > len(self):
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with length {len(self)}"
            )
        # memoryview's implementation is very fast but not always possible (half float, fixed-size binary, interval)
        if self._data_type in (
            NANOARROW_TYPE_HALF_FLOAT,
            NANOARROW_TYPE_INTERVAL_DAY_TIME,
            NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO,
            NANOARROW_TYPE_DECIMAL128,
            NANOARROW_TYPE_DECIMAL256
        ) or (
            self._data_type == NANOARROW_TYPE_BINARY and self._element_size_bits != 0
        ):
            return self._iter_struct(offset, length)
        else:
            return self._iter_memoryview(offset, length)

    def _iter_memoryview(self, int64_t offset, int64_t length):
        return iter(memoryview(self)[offset:(offset + length)])

    def _iter_struct(self, int64_t offset, int64_t length):
        for value in iter_unpack(self.format, self):
            if len(value) == 1:
                yield value[0]
            else:
                yield value

    @property
    def n_elements(self):
        return self._n_elements

    def element(self, i):
        if self._data_type == NANOARROW_TYPE_BOOL:
            if i < 0 or i >= self.n_elements:
                raise IndexError(f"Index {i} out of range")
            return ArrowBitGet(self._ptr.data.as_uint8, i)
        else:
            return self[i]

    def elements(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        if offset < 0 or length < 0 or (offset + length) > self.n_elements:
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with {self.n_elements} elements"
            )

        if self._data_type == NANOARROW_TYPE_BOOL:
            return self._iter_bitmap(offset, length)
        else:
            return self._iter_dispatch(offset, length)

    def copy_into(self, dest, offset=0, length=None, dest_offset=0):
        if length is None:
            length = self.n_elements

        cdef Py_buffer buffer
        PyObject_GetBuffer(dest, &buffer, PyBUF_WRITABLE | PyBUF_ANY_CONTIGUOUS)

        cdef int64_t c_offset = offset
        cdef int64_t c_length = length
        cdef int64_t c_item_size = self.item_size
        cdef int64_t c_dest_offset = dest_offset
        self._check_copy_into_bounds(&buffer, c_offset, c_length, dest_offset, c_item_size)

        cdef uint8_t* dest_uint8 = <uint8_t*>buffer.buf
        cdef int64_t dest_offset_bytes = c_dest_offset * c_item_size
        cdef int64_t src_offset_bytes = c_offset * c_item_size
        cdef int64_t bytes_to_copy = c_length * c_item_size

        memcpy(
            &(dest_uint8[dest_offset_bytes]),
            &(self._ptr.data.as_uint8[src_offset_bytes]),
            bytes_to_copy
        )

        PyBuffer_Release(&buffer)
        return bytes_to_copy

    def unpack_bits_into(self, dest, offset=0, length=None, dest_offset=0):
        if self._data_type != NANOARROW_TYPE_BOOL:
            raise ValueError("Can't unpack non-boolean buffer")

        if length is None:
            length = self.n_elements

        cdef Py_buffer buffer
        PyObject_GetBuffer(dest, &buffer, PyBUF_WRITABLE | PyBUF_ANY_CONTIGUOUS)
        self._check_copy_into_bounds(&buffer, offset, length, dest_offset, 1)

        ArrowBitsUnpackInt8(
            self._ptr.data.as_uint8,
            offset,
            length,
            &(<int8_t*>buffer.buf)[dest_offset]
        )

        PyBuffer_Release(&buffer)
        return length

    def unpack_bits(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        out = CBufferBuilder().set_format("?")
        out.reserve_bytes(length)
        self.unpack_bits_into(out, offset, length)
        out.advance(length)
        return out.finish()

    def copy(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        cdef int64_t bytes_to_copy = length * self.item_size
        out = CBufferBuilder().set_data_type(self.data_type_id)
        out.reserve_bytes(bytes_to_copy)
        self.copy_into(out, offset, length)
        out.advance(bytes_to_copy)
        return out.finish()

    cdef _check_copy_into_bounds(self, Py_buffer* dest, int64_t offset, int64_t length,
                                 int64_t dest_offset, int64_t dest_itemsize):
        if offset < 0 or length < 0 or (offset + length) > self.n_elements:
            PyBuffer_Release(dest)
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with {self.n_elements} elements"
            )

        if dest.itemsize != 1 and dest.itemsize != dest_itemsize:
            raise ValueError(
                "Destination buffer must have itemsize == 1 or "
                f"itemsize == {dest_itemsize}"
            )

        cdef int64_t dest_offset_bytes = dest_offset * dest_itemsize
        cdef int64_t bytes_to_copy = dest_itemsize * length
        if dest_offset < 0 or dest.len < (dest_offset_bytes + bytes_to_copy):
            buffer_len = dest.len
            PyBuffer_Release(dest)
            raise IndexError(
                f"Can't unpack {length} elements into buffer of size {buffer_len} "
                f"with dest_offset = {dest_offset}"
            )

    def _iter_bitmap(self, int64_t offset, int64_t length):
        cdef uint8_t item
        cdef int64_t i

        if offset % 8 == 0:
            first_byte = offset // 8
            last_byte = self._shape
            i = 0

            for byte_i in range(first_byte, last_byte):
                item = self._ptr.data.as_uint8[byte_i]
                for j in range(8):
                    yield (item & (<uint8_t>1 << j)) != 0
                    i += 1
                    if i >= length:
                        return
        else:
            for i in range(length):
                yield ArrowBitGet(self._ptr.data.as_uint8, offset + i) != 0


    cdef Py_ssize_t _item_size(self):
        if self._element_size_bits < 8:
            return 1
        else:
            return self._element_size_bits // 8


    def __dlpack__(self):
        # Note: parent offset not applied!
        return view_to_dlpack(self)


    def __dlpack_device__(self):
        return (view_to_dlpack_device(self).device_type,
                view_to_dlpack_device(self).device_id)


    # These are special methods, which can't be cdef and we can't
    # call them from elsewhere. We implement the logic for the buffer
    # protocol separately so we can re-use it in the CBuffer.
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._do_getbuffer(buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        self._do_releasebuffer(buffer)

    cdef _do_getbuffer(self, Py_buffer *buffer, int flags):
        if self._device is not DEVICE_CPU:
            raise RuntimeError("CBufferView is not a CPU buffer")

        if flags & PyBUF_WRITABLE:
            raise BufferError("CBufferView does not support PyBUF_WRITABLE")

        buffer.buf = <void*>self._ptr.data.data

        if flags & PyBUF_FORMAT:
            buffer.format = self._format
        else:
            buffer.format = NULL

        buffer.internal = NULL
        buffer.itemsize = self._strides
        buffer.len = self._ptr.size_bytes
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = &self._shape
        buffer.strides = &self._strides
        buffer.suboffsets = NULL

    cdef _do_releasebuffer(self, Py_buffer* buffer):
        pass

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_lib")
        return f"{class_label}({_repr_utils.buffer_view_repr(self)})"


cdef class CBuffer:
    """Wrapper around readable owned buffer content

    Like the CBufferView, the CBuffer represents readable buffer content; however,
    unlike the CBufferView, the CBuffer always represents a valid ArrowBuffer C object.
    """
    cdef object _base
    cdef ArrowBuffer* _ptr
    cdef ArrowType _data_type
    cdef int _element_size_bits
    cdef char _format[32]
    cdef Device _device
    cdef CBufferView _view
    cdef int _get_buffer_count

    def __cinit__(self):
        self._base = None
        self._ptr = NULL
        self._data_type = NANOARROW_TYPE_BINARY
        self._element_size_bits = 0
        self._device = DEVICE_CPU
        # Set initial format to "B" (Cython makes this hard)
        self._format[0] = 66
        self._format[1] = 0
        self._get_buffer_count = 0
        self._view = CBufferView(None, 0, 0, NANOARROW_TYPE_BINARY, 0, self._device)

    cdef _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("CBuffer is not valid")

    cdef _assert_buffer_count_zero(self):
        if self._get_buffer_count != 0:
            raise RuntimeError(
                f"CBuffer already open ({self._get_buffer_count} ",
                f"references, {self._writable_get_buffer_count} writable)")

    cdef _populate_view(self):
        self._assert_valid()
        self._assert_buffer_count_zero()
        self._view = CBufferView(
            self._base, <uintptr_t>self._ptr.data,
            self._ptr.size_bytes, self._data_type, self._element_size_bits,
            self._device
        )

        snprintf(self._view._format, sizeof(self._view._format), "%s", self._format)

    @staticmethod
    def empty():
        cdef CBuffer out = CBuffer()
        out._base = alloc_c_buffer(&out._ptr)
        return out

    @staticmethod
    def from_pybuffer(obj):
        cdef CBuffer out = CBuffer()
        out._base = alloc_c_buffer(&out._ptr)
        out._set_format(c_buffer_set_pybuffer(obj, &out._ptr))
        out._device = DEVICE_CPU
        out._populate_view()
        return out

    def _set_format(self, str format):
        self._assert_buffer_count_zero()

        element_size_bytes, data_type = c_arrow_type_from_format(format)
        self._data_type = data_type
        self._element_size_bits = element_size_bytes * 8
        format_bytes = format.encode("UTF-8")
        snprintf(self._format, sizeof(self._format), "%s", <const char*>format_bytes)
        self._populate_view()
        return self

    def _set_data_type(self, ArrowType type_id, int element_size_bits=0):
        self._assert_buffer_count_zero()

        self._element_size_bits = c_format_from_arrow_type(
            type_id,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._data_type = type_id

        self._populate_view()
        return self

    def _addr(self):
        self._assert_valid()
        return <uintptr_t>self._ptr.data

    @property
    def _get_buffer_count(self):
        return self._get_buffer_count

    @property
    def size_bytes(self):
        self._assert_valid()
        return self._ptr.size_bytes

    @property
    def data_type(self):
        return ArrowTypeString(self._data_type).decode("UTF-8")

    @property
    def data_type_id(self):
        return self._data_type

    @property
    def element_size_bits(self):
        return self._element_size_bits

    @property
    def item_size(self):
        self._assert_valid()
        return self._view.item_size

    @property
    def format(self):
        return self._format.decode("UTF-8")

    def __len__(self):
        self._assert_valid()
        return len(self._view)

    def __getitem__(self, k):
        self._assert_valid()
        return self._view[k]

    def __iter__(self):
        self._assert_valid()
        return iter(self._view)

    @property
    def n_elements(self):
        self._assert_valid()
        return self._view.n_elements

    def element(self, i):
        self._assert_valid()
        return self._view.element(i)

    def elements(self, offset=0, length=None):
        self._assert_valid()
        return self._view.elements(offset, length)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self._assert_valid()
        self._view._do_getbuffer(buffer, flags)
        self._get_buffer_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        if self._get_buffer_count <= 0:
            raise RuntimeError("CBuffer buffer reference count underflow (releasebuffer)")

        self._view._do_releasebuffer(buffer)
        self._get_buffer_count -= 1

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_lib")
        if self._ptr == NULL:
            return f"{class_label}(<invalid>)"

        return f"{class_label}({_repr_utils.buffer_view_repr(self._view)})"


cdef class CBufferBuilder:
    """Wrapper around writable CPU buffer content

    This class provides a growable type-aware buffer that can be used
    to create a typed buffer from a Python iterable. This method of
    creating a buffer is usually slower than constructors like
    ``array.array()`` or ``numpy.array()``; however, this class supports
    all Arrow types with a single data buffer (e.g., boolean bitmaps,
    float16, intervals, fixed-size binary), some of which are not supported
    by other constructors.
    """
    cdef CBuffer _buffer
    cdef bint _locked

    def __cinit__(self):
        self._buffer = CBuffer.empty()
        self._locked = False

    cdef _assert_unlocked(self):
        if self._locked:
            raise BufferError("CBufferBuilder is locked")

    # Implement the buffer protocol so that this object can be used as
    # the argument to Struct.readinto() (or perhaps written to by
    # an independent library).
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self._assert_unlocked()
        PyBuffer_FillInfo(
            buffer,
            self,
            self._buffer._ptr.data,
            self._buffer._ptr.capacity_bytes,
            0,
            flags
        )
        self._locked = True

    def __releasebuffer__(self, Py_buffer* buffer):
        self._locked = False

    def set_data_type(self, ArrowType type_id, int element_size_bits=0):
        """Set the data type used to interpret elements in :meth:`write_elements`."""
        self._buffer._set_data_type(type_id, element_size_bits)
        return self

    def set_format(self, str format):
        """Set the Python buffer format used to interpret elements in
        :meth:`write_elements`.
        """
        self._buffer._set_format(format)
        return self

    @property
    def format(self):
        """The ``struct`` format code of the underlying buffer"""
        return self._buffer._format.decode()

    @property
    def size_bytes(self):
        """The number of bytes that have been written to this buffer"""
        return self._buffer.size_bytes

    @property
    def capacity_bytes(self):
        """The number of bytes allocated in the underlying buffer"""
        return self._buffer._ptr.capacity_bytes

    def reserve_bytes(self, int64_t additional_bytes):
        """Ensure that the underlying buffer has space for ``additional_bytes``
        more bytes to be written"""
        self._assert_unlocked()
        cdef int code = ArrowBufferReserve(self._buffer._ptr, additional_bytes)
        Error.raise_error_not_ok("ArrowBufferReserve()", code)
        return self

    def advance(self, int64_t additional_bytes):
        """Manually increase :attr:`size_bytes` by ``additional_bytes``

        This can be used after writing to the buffer using the buffer protocol
        to ensure that :attr:`size_bytes` accurately reflects the number of
        bytes written to the buffer.
        """
        cdef int64_t new_size = self._buffer._ptr.size_bytes + additional_bytes
        if new_size < 0 or new_size > self._buffer._ptr.capacity_bytes:
            raise IndexError(f"Can't advance {additional_bytes} from {self.size_bytes}")

        self._buffer._ptr.size_bytes = new_size
        return self

    def write(self, content):
        """Write bytes to this buffer

        Writes the bytes of ``content`` without considering the element type of
        ``content`` or the element type of this buffer.

        This method returns the number of bytes that were written.
        """
        self._assert_unlocked()

        cdef Py_buffer buffer
        cdef int64_t out
        PyObject_GetBuffer(content, &buffer, PyBUF_ANY_CONTIGUOUS)

        # TODO: Check for single dimension?

        cdef int code = ArrowBufferReserve(self._buffer._ptr, buffer.len)
        if code != NANOARROW_OK:
            PyBuffer_Release(&buffer)
            Error.raise_error("ArrowBufferReserve()", code)

        code = PyBuffer_ToContiguous(
            self._buffer._ptr.data + self._buffer._ptr.size_bytes,
            &buffer,
            buffer.len,
            # 'C' (not sure how to pass a character literal here)
            43
        )
        out = buffer.len
        PyBuffer_Release(&buffer)
        Error.raise_error_not_ok("PyBuffer_ToContiguous()", code)

        self._buffer._ptr.size_bytes += out
        return out

    def write_elements(self, obj):
        """"Write an iterable of elements to this buffer

        Writes the elements of iterable ``obj`` according to the binary
        representation specified by :attr:`format`. This is currently
        powered by ``struct.pack_into()`` except when building bitmaps
        where an internal implementation is used.

        This method returns the number of elements that were written.
        """
        self._assert_unlocked()

        # Boolean arrays need their own writer since Python provides
        # no helpers for bitpacking
        if self._buffer._data_type == NANOARROW_TYPE_BOOL:
            return self._write_bits(obj)

        cdef int64_t n_values = 0
        cdef int64_t bytes_per_element = calcsize(self._buffer.format)
        cdef int code

        struct_obj = Struct(self._buffer._format)
        pack = struct_obj.pack
        write = self.write

        # If an object has a length, we can avoid extra allocations
        if hasattr(obj, "__len__"):
            code = ArrowBufferReserve(self._buffer._ptr, bytes_per_element * len(obj))
            Error.raise_error_not_ok("ArrowBufferReserve()", code)

        # Types whose Python representation is a tuple need a slightly different
        # pack_into() call. if code != NANOARROW_OK is used instead of
        # Error.raise_error_not_ok() Cython can avoid the extra function call
        # and this is a very tight loop.
        if self._buffer._data_type in (NANOARROW_TYPE_INTERVAL_DAY_TIME,
                                       NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO):
            for item in obj:
                write(pack(*item))
                n_values += 1

        else:
            for item in obj:
                write(pack(item))
                n_values += 1

        return n_values

    cdef _write_bits(self, obj):
        if self._buffer._ptr.size_bytes != 0:
            raise NotImplementedError("Append to bitmap that has already been appended to")

        cdef char buffer_item = 0
        cdef int buffer_item_i = 0
        cdef int code
        cdef int64_t n_values = 0
        for item in obj:
            n_values += 1
            if item:
                buffer_item |= (<char>1 << buffer_item_i)

            buffer_item_i += 1
            if buffer_item_i == 8:
                code = ArrowBufferAppendInt8(self._buffer._ptr, buffer_item)
                Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)
                buffer_item = 0
                buffer_item_i = 0

        if buffer_item_i != 0:
            code = ArrowBufferAppendInt8(self._buffer._ptr, buffer_item)
            Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)

        return n_values

    def finish(self):
        """Finish building this buffer

        Performs any steps required to finish building this buffer and
        returns the result. Any behaviour resulting from calling methods
        on this object after it has been finished is not currently
        defined (but should not crash).
        """
        self._assert_unlocked()

        cdef CBuffer out = self._buffer
        out._populate_view()

        self._buffer = CBuffer.empty()
        return out

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_lib")
        return f"{class_label}({self.size_bytes}/{self.capacity_bytes})"


cdef class NoneAwareWrapperIterator:
    """Nullable iterator wrapper

    This class wraps an iterable ``obj`` that might contain ``None`` values
    such that the iterable provided by this class contains "empty" (but valid)
    values. After ``obj`` has been completely consumed, one can call
    ``finish()`` to obtain the resulting bitmap. This is useful for passing
    iterables that might contain None to tools that cannot handle them
    (e.g., struct.pack(), array.array()).
    """
    cdef ArrowBitmap _bitmap
    cdef object _obj
    cdef object _value_if_none
    cdef int64_t _valid_count
    cdef int64_t _item_count

    def __cinit__(self, obj, type_id, item_size_bytes=0):
        ArrowBitmapInit(&self._bitmap)
        self._obj = iter(obj)

        self._value_if_none = self._get_value_if_none(type_id, item_size_bytes)
        self._valid_count = 0
        self._item_count = 0

    def __dealloc__(self):
        ArrowBitmapReset(&self._bitmap)

    def reserve(self, int64_t additional_elements):
        cdef int code = ArrowBitmapReserve(&self._bitmap, additional_elements)
        Error.raise_error_not_ok(self, code)

    def _get_value_if_none(self, type_id, item_size_bytes=0):
        if type_id == NANOARROW_TYPE_INTERVAL_DAY_TIME:
            return (0, 0)
        elif type_id == NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
            return (0, 0, 0)
        elif type_id == NANOARROW_TYPE_BOOL:
            return False
        elif type_id  in (NANOARROW_TYPE_BINARY, NANOARROW_TYPE_FIXED_SIZE_BINARY):
            return b"\x00" * item_size_bytes
        elif type_id in (NANOARROW_TYPE_HALF_FLOAT, NANOARROW_TYPE_FLOAT, NANOARROW_TYPE_DOUBLE):
            return 0.0
        else:
            return 0

    cdef _append_to_validity(self, int is_valid):
        self._valid_count += is_valid
        self._item_count += 1

        # Avoid allocating a bitmap if all values seen so far are valid
        if self._valid_count == self._item_count:
            return

        # If the bitmap hasn't been allocated yet, allocate it now and
        # fill with 1s for all previous elements.
        cdef int code
        if self._bitmap.size_bits == 0 and self._item_count > 1:
            code = ArrowBitmapAppend(&self._bitmap, 1, self._item_count - 1)
            if code != NANOARROW_OK:
                Error.raise_error("ArrowBitmapAppend()", code)

        # Append this element to the bitmap
        code = ArrowBitmapAppend(&self._bitmap, is_valid, 1)
        if code != NANOARROW_OK:
            Error.raise_error("ArrowBitmapAppend()", code)

    def __iter__(self):
        for item in self._obj:
            if item is None:
                self._append_to_validity(0)
                yield self._value_if_none
            else:
                self._append_to_validity(1)
                yield item

    def finish(self):
        """Obtain the total count, null count, and validity bitmap after
        consuming this iterable."""
        cdef CBuffer validity
        null_count = self._item_count - self._valid_count

        # If we did allocate a bitmap, make sure the last few bits are zeroed
        if null_count > 0 and self._bitmap.size_bits % 8 != 0:
            ArrowBitmapAppendUnsafe(&self._bitmap, 0, self._bitmap.size_bits % 8)

        if null_count > 0:
            validity = CBuffer.empty()
            ArrowBufferMove(&self._bitmap.buffer, validity._ptr)
            validity._set_data_type(NANOARROW_TYPE_BOOL)

        return (
            self._item_count,
            null_count,
            validity if null_count > 0 else None
        )


cdef class CArrayBuilder:
    cdef CArray c_array
    cdef ArrowArray* _ptr
    cdef bint _can_validate

    def __cinit__(self, CArray array):
        self.c_array = array
        self._ptr = array._ptr
        self._can_validate = True

    @staticmethod
    def allocate():
        return CArrayBuilder(CArray.allocate(CSchema.allocate()))

    def is_empty(self):
        if self._ptr.release == NULL:
            raise RuntimeError("CArrayBuilder is not initialized")

        return self._ptr.length == 0

    def init_from_type(self, int type_id):
        if self._ptr.release != NULL:
            raise RuntimeError("CArrayBuilder is already initialized")

        cdef int code = ArrowArrayInitFromType(self._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowArrayInitFromType()", code)

        code = ArrowSchemaInitFromType(self.c_array._schema._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowSchemaInitFromType()", code)

        return self

    def init_from_schema(self, CSchema schema):
        if self._ptr.release != NULL:
            raise RuntimeError("CArrayBuilder is already initialized")

        cdef Error error = Error()
        cdef int code = ArrowArrayInitFromSchema(self._ptr, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayInitFromType()", code)

        self.c_array._schema = schema
        return self

    def start_appending(self):
        cdef int code = ArrowArrayStartAppending(self._ptr)
        Error.raise_error_not_ok("ArrowArrayStartAppending()", code)
        return self

    def append_strings(self, obj):
        cdef int code
        cdef Py_ssize_t item_utf8_size
        cdef ArrowStringView item

        for py_item in obj:
            if py_item is None:
                code = ArrowArrayAppendNull(self._ptr, 1)
            else:
                # Cython raises the error from PyUnicode_AsUTF8AndSize()
                # in the event that py_item is not a str(); however, we
                # set item_utf8_size = 0 to be safe.
                item_utf8_size = 0
                item.data = PyUnicode_AsUTF8AndSize(py_item, &item_utf8_size)
                item.size_bytes = item_utf8_size
                code = ArrowArrayAppendString(self._ptr, item)

            if code != NANOARROW_OK:
                Error.raise_error(f"append string item {py_item}")

        return self

    def append_bytes(self, obj):
        cdef Py_buffer buffer
        cdef ArrowBufferView item

        for py_item in obj:
            if py_item is None:
                code = ArrowArrayAppendNull(self._ptr, 1)
            else:
                PyObject_GetBuffer(py_item, &buffer, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT)

                if buffer.ndim != 1:
                    raise ValueError("Can't append buffer with dimensions != 1 to binary array")

                if buffer.itemsize != 1:
                    PyBuffer_Release(&buffer)
                    raise ValueError("Can't append buffer with itemsize != 1 to binary array")

                item.data.data = buffer.buf
                item.size_bytes = buffer.len
                code = ArrowArrayAppendBytes(self._ptr, item)
                PyBuffer_Release(&buffer)

            if code != NANOARROW_OK:
                Error.raise_error(f"append bytes item {py_item}")

    def set_offset(self, int64_t offset):
        self.c_array._assert_valid()
        self._ptr.offset = offset
        return self

    def set_length(self, int64_t length):
        self.c_array._assert_valid()
        self._ptr.length = length
        return self

    def set_null_count(self, int64_t null_count):
        self.c_array._assert_valid()
        self._ptr.null_count = null_count
        return self

    def resolve_null_count(self):
        self.c_array._assert_valid()

        # This doesn't apply to unions. We currently don't have a schema view
        # or array view we can use to query the type ID, so just use the format
        # string for now.
        format = self.c_array.schema.format
        if format.startswith("+us:") or format.startswith("+ud:"):
            return self

        # Don't overwrite an explicit null count
        if self._ptr.null_count != -1:
            return self

        cdef ArrowBuffer* validity_buffer = ArrowArrayBuffer(self._ptr, 0)
        if validity_buffer.size_bytes == 0:
            self._ptr.null_count = 0
            return self

        # From _ArrowBytesForBits(), which is not included in nanoarrow_c.pxd
        # because it's an internal inline function.
        cdef int64_t bits = self._ptr.offset + self._ptr.length
        cdef int64_t bytes_required = (bits >> 3) + ((bits & 7) != 0)

        if validity_buffer.size_bytes < bytes_required:
            raise ValueError(
                f"Expected validity bitmap >= {bytes_required} bytes "
                f"but got validity bitmap with {validity_buffer.size_bytes} bytes"
            )

        cdef int64_t count = ArrowBitCountSet(
            validity_buffer.data,
            self._ptr.offset,
            self._ptr.length
        )
        self._ptr.null_count = self._ptr.length - count
        return self

    def set_buffer(self, int64_t i, CBuffer buffer, move=False):
        """Sets a buffer of this ArrowArray such the pointer at array->buffers[i] is
        equal to buffer->data and such that the buffer's lifcycle is managed by
        the array. If move is True, the input Python object that previously wrapped
        the ArrowBuffer will be invalidated, which is usually the desired behaviour
        if you built or imported a buffer specifically to build this array. If move
        is False (the default), this function will a make a shallow copy via another
        layer of Python object wrapping."""
        if i < 0 or i > 3:
            raise IndexError("i must be >= 0 and <= 3")

        self.c_array._assert_valid()
        if not move:
            buffer = CBuffer.from_pybuffer(buffer)

        ArrowBufferMove(buffer._ptr, ArrowArrayBuffer(self._ptr, i))

        # The buffer's lifecycle is now owned by the array; however, we need
        # array->buffers[i] to be updated such that it equals
        # ArrowArrayBuffer(array, i)->data.
        self._ptr.buffers[i] = ArrowArrayBuffer(self._ptr, i).data

        return self

    def set_child(self, int64_t i, CArray c_array, move=False):
        cdef CArray child = self.c_array.child(i)
        if child._ptr.release != NULL:
            ArrowArrayRelease(child._ptr)

        if not move:
            c_array_shallow_copy(c_array._base, c_array._ptr, child._ptr)
        else:
            ArrowArrayMove(c_array._ptr, child._ptr)

        # After setting children, we can't use the built-in validation done by
        # ArrowArrayFinishBuilding() because it assumes that the private_data of
        # each array (recursively) is one that was initialized by ArrowArrayInit()
        self._can_validate = False

        return self

    def finish(self, validation_level=None):
        self.c_array._assert_valid()
        cdef ArrowValidationLevel c_validation_level
        cdef Error error = Error()
        cdef int code

        if self._can_validate:
            c_validation_level = NANOARROW_VALIDATION_LEVEL_DEFAULT
            if validation_level == "full":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_FULL
            elif validation_level == "minimal":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_MINIMAL
            elif validation_level == "none":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_NONE

            code = ArrowArrayFinishBuilding(self._ptr, c_validation_level, &error.c_error)
            error.raise_message_not_ok("ArrowArrayFinishBuildingDefault()", code)

        elif validation_level not in (None, "none"):
            raise NotImplementedError("Validation for array with children is not implemented")

        out = self.c_array
        self.c_array = CArray.allocate(CSchema.allocate())
        self._ptr = self.c_array._ptr
        self._can_validate = True

        return out


cdef class CArrayStream:
    """Low-level ArrowArrayStream wrapper

    This object is a literal wrapper around an ArrowArrayStream. It provides methods that
    that wrap the underlying C callbacks and handles the C Data interface lifecycle
    (i.e., initialized ArrowArrayStream structures are always released).

    See `nanoarrow.c_array_stream()` for construction and usage examples.
    """
    cdef object _base
    cdef ArrowArrayStream* _ptr
    cdef object _cached_schema

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayStream*>addr
        self._cached_schema = None

    @staticmethod
    def allocate():
        cdef ArrowArrayStream* c_array_stream_out
        base = alloc_c_array_stream(&c_array_stream_out)
        return CArrayStream(base, <uintptr_t>c_array_stream_out)

    @staticmethod
    def from_c_arrays(arrays, CSchema schema, move=False, validate=True):
        cdef ArrowArrayStream* c_array_stream_out
        base = alloc_c_array_stream(&c_array_stream_out)

        # Don't create more copies than we have to (but make sure
        # one exists for validation if requested)
        cdef CSchema out_schema = schema
        if validate and not move:
            validate_schema = schema
            out_schema = schema.__deepcopy__()
        elif validate:
            validate_schema = schema.__deepcopy__()
            out_schema = schema
        elif not move:
            out_schema = schema.__deepcopy__()

        cdef int code = ArrowBasicArrayStreamInit(c_array_stream_out, out_schema._ptr, len(arrays))
        Error.raise_error_not_ok("ArrowBasicArrayStreamInit()", code)

        cdef ArrowArray tmp
        cdef CArray array
        for i in range(len(arrays)):
            array = arrays[i]

            if validate:
                assert_type_equal(array.schema, validate_schema)

            if not move:
                c_array_shallow_copy(array._base, array._ptr, &tmp)
                ArrowBasicArrayStreamSetArray(c_array_stream_out, i, &tmp)
            else:
                ArrowBasicArrayStreamSetArray(c_array_stream_out, i, array._ptr)

        cdef Error error = Error()
        if validate:
            code = ArrowBasicArrayStreamValidate(c_array_stream_out, &error.c_error)
            error.raise_message_not_ok("ArrowBasicArrayStreamValidate()", code)

        return CArrayStream(base, <uintptr_t>c_array_stream_out)

    def release(self):
        if self.is_valid():
            self._ptr.release(self._ptr)

    @staticmethod
    def _import_from_c_capsule(stream_capsule):
        """
        Import from a ArrowArrayStream PyCapsule.

        Parameters
        ----------
        stream_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_array_stream' containing an
            ArrowArrayStream pointer.
        """
        return CArrayStream(
            stream_capsule,
            <uintptr_t>PyCapsule_GetPointer(stream_capsule, 'arrow_array_stream')
        )

    def __arrow_c_stream__(self, requested_schema=None):
        """
        Export the stream as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        PyCapsule
        """
        self._assert_valid()

        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        cdef:
            ArrowArrayStream* c_array_stream_out

        array_stream_capsule = alloc_c_array_stream(&c_array_stream_out)
        ArrowArrayStreamMove(self._ptr, c_array_stream_out)
        return array_stream_capsule

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("array stream pointer is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("array stream is released")

    def _get_schema(self, CSchema schema):
        self._assert_valid()
        cdef Error error = Error()
        cdef int code = ArrowArrayStreamGetSchema(self._ptr, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayStream::get_schema()", code)

    def _get_cached_schema(self):
        if self._cached_schema is None:
            self._cached_schema = CSchema.allocate()
            self._get_schema(self._cached_schema)

        return self._cached_schema

    def get_schema(self):
        """Get the schema associated with this stream
        """
        out = CSchema.allocate()
        self._get_schema(out)
        return out

    def get_next(self):
        """Get the next Array from this stream

        Raises StopIteration when there are no more arrays in this stream.
        """
        self._assert_valid()

        # We return a reference to the same Python object for each
        # Array that is returned. This is independent of get_schema(),
        # which is guaranteed to call the C object's callback and
        # faithfully pass on the returned value.

        cdef Error error = Error()
        cdef CArray array = CArray.allocate(self._get_cached_schema())
        cdef int code = ArrowArrayStreamGetNext(self._ptr, array._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayStream::get_next()", code)

        if not array.is_valid():
            raise StopIteration()
        else:
            return array

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.release()

    def __repr__(self):
        return _repr_utils.array_stream_repr(self)


cdef class CMaterializedArrayStream:
    cdef CSchema _schema
    cdef CBuffer _array_ends
    cdef list _arrays
    cdef int64_t _total_length

    def __cinit__(self):
        self._arrays = []
        self._total_length = 0
        self._schema = CSchema.allocate()
        self._array_ends = CBuffer.empty()
        cdef int code = ArrowBufferAppendInt64(self._array_ends._ptr, 0)
        Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)

    cdef _finalize(self):
        self._array_ends._set_data_type(NANOARROW_TYPE_INT64)

    @property
    def schema(self):
        return self._schema

    def __getitem__(self, k):
        cdef int64_t kint
        cdef int array_i
        cdef const int64_t* sorted_offsets = <int64_t*>self._array_ends._ptr.data

        if isinstance(k, slice):
            raise NotImplementedError("index with slice")

        kint = k
        if kint < 0:
            kint += self._total_length
        if kint < 0 or kint >= self._total_length:
            raise IndexError(f"Index {kint} is out of range")

        array_i = ArrowResolveChunk64(kint, sorted_offsets, 0, len(self._arrays))
        kint -= sorted_offsets[array_i]
        return self._arrays[array_i], kint

    def __len__(self):
        return self._array_ends[len(self._arrays)]

    def __iter__(self):
        for c_array in self._arrays:
            for item_i in range(len(c_array)):
                yield c_array, item_i

    def array(self, int64_t i):
        return self._arrays[i]

    @property
    def n_arrays(self):
        return len(self._arrays)

    @property
    def arrays(self):
        return iter(self._arrays)

    def __arrow_c_stream__(self, requested_schema=None):
        # When an array stream from iterable is supported, that could be used here
        # to avoid unnessary shallow copies.
        stream = CArrayStream.from_c_arrays(
            self._arrays,
            self._schema,
            move=False,
            validate=False
        )

        return stream.__arrow_c_stream__(requested_schema=requested_schema)

    def child(self, int64_t i):
        cdef CMaterializedArrayStream out = CMaterializedArrayStream()
        cdef int code

        out._schema = self._schema.child(i)
        out._arrays = [chunk.child(i) for chunk in self._arrays]
        for child_chunk in out._arrays:
            out._total_length += len(child_chunk)
            code = ArrowBufferAppendInt64(out._array_ends._ptr, out._total_length)
            Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)

        out._finalize()
        return out

    @staticmethod
    def from_c_arrays(arrays, CSchema schema, bint validate=True):
        cdef CMaterializedArrayStream out = CMaterializedArrayStream()

        for array in arrays:
            if not isinstance(array, CArray):
                raise TypeError(f"Expected CArray but got {type(array).__name__}")

            if len(array) == 0:
                continue

            if validate:
                assert_type_equal(array.schema, schema)

            out._total_length += len(array)
            code = ArrowBufferAppendInt64(out._array_ends._ptr, out._total_length)
            Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)
            out._arrays.append(array)

        out._schema = schema
        out._finalize()
        return out

    @staticmethod
    def from_c_array(CArray array):
        return CMaterializedArrayStream.from_c_arrays(
            [array],
            array.schema,
            validate=False
        )

    @staticmethod
    def from_c_array_stream(CArrayStream stream):
        with stream:
            return CMaterializedArrayStream.from_c_arrays(
                stream,
                stream._get_cached_schema(),
                validate=False
            )


cdef class CDeviceArray:
    cdef object _base
    cdef ArrowDeviceArray* _ptr
    cdef CSchema _schema

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowDeviceArray*>addr
        self._schema = schema

    @property
    def schema(self):
        return self._schema

    @property
    def device_type(self):
        return DeviceType(self._ptr.device_type)

    @property
    def device_type_id(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @property
    def array(self):
        # TODO: We lose access to the sync_event here, so we probably need to
        # synchronize (or propagate it, or somehow prevent data access downstream)
        cdef CArray array = CArray(self, <uintptr_t>&self._ptr.array, self._schema)
        array._set_device(self._ptr.device_type, self._ptr.device_id)
        return array

    def view(self):
        return self.array.view()

    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema=requested_schema)

    def __arrow_c_device_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        # TODO: evaluate whether we need to synchronize here or whether we should
        # move device arrays instead of shallow-copying them
        cdef ArrowDeviceArray* c_array_out
        device_array_capsule = alloc_c_device_array(&c_array_out)
        c_device_array_shallow_copy(self._base, self._ptr, c_array_out)

        return self._schema.__arrow_c_schema__(), device_array_capsule

    @staticmethod
    def _import_from_c_capsule(schema_capsule, device_array_capsule):
        """
        Import from an ArrowSchema and ArrowArray PyCapsule tuple.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        device_array_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_device_array' containing an
            ArrowDeviceArray pointer.
        """
        cdef:
            CSchema out_schema
            CDeviceArray out

        out_schema = CSchema._import_from_c_capsule(schema_capsule)
        out = CDeviceArray(
            device_array_capsule,
            <uintptr_t>PyCapsule_GetPointer(device_array_capsule, 'arrow_device_array'),
            out_schema
        )

        return out

    def __repr__(self):
        return _repr_utils.device_array_repr(self)
