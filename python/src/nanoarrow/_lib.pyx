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
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_IsValid
from cpython cimport (
    Py_buffer,
    PyObject_CheckBuffer,
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBuffer_ToContiguous,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_FORMAT,
    PyBUF_WRITABLE
)
from cpython.ref cimport Py_INCREF, Py_DECREF
from nanoarrow_c cimport *
from nanoarrow_device_c cimport *

from sys import byteorder as sys_byteorder
from struct import unpack_from, iter_unpack, calcsize, Struct
from nanoarrow import _lib_utils

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


cdef object alloc_c_schema(ArrowSchema** c_schema) noexcept:
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


cdef object alloc_c_array(ArrowArray** c_array) noexcept:
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


cdef object alloc_c_array_stream(ArrowArrayStream** c_stream) noexcept:
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


cdef object alloc_c_device_array(ArrowDeviceArray** c_device_array) noexcept:
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


cdef object alloc_c_array_view(ArrowArrayView** c_array_view) noexcept:
    c_array_view[0] = <ArrowArrayView*> ArrowMalloc(sizeof(ArrowArrayView))
    ArrowArrayViewInitFromType(c_array_view[0], NANOARROW_TYPE_UNINITIALIZED)
    return PyCapsule_New(c_array_view[0], 'nanoarrow_array_view', &pycapsule_array_view_deleter)


cdef void arrow_array_release(ArrowArray* array) noexcept with gil:
    Py_DECREF(<object>array.private_data)
    array.private_data = NULL
    array.release = NULL


cdef object alloc_c_array_shallow_copy(object base, const ArrowArray* c_array) noexcept:
    """Make a shallow copy of an ArrowArray

    To more safely implement export of an ArrowArray whose address may be
    depended on by some other Python object, we implement a shallow copy
    whose constructor calls Py_INCREF() on a Python object responsible
    for the ArrowArray's lifecycle and whose deleter calls Py_DECREF() on
    the same object.
    """
    cdef:
        ArrowArray* c_array_out

    array_capsule = alloc_c_array(&c_array_out)

    # shallow copy
    memcpy(c_array_out, c_array, sizeof(ArrowArray))
    c_array_out.release = NULL
    c_array_out.private_data = NULL

    # track original base
    c_array_out.private_data = <void*>base
    Py_INCREF(base)
    c_array_out.release = arrow_array_release

    return array_capsule


cdef void pycapsule_buffer_deleter(object stream_capsule) noexcept:
    cdef ArrowBuffer* buffer = <ArrowBuffer*>PyCapsule_GetPointer(
        stream_capsule, 'nanoarrow_buffer'
    )

    ArrowBufferReset(buffer)
    ArrowFree(buffer)


cdef object alloc_c_buffer(ArrowBuffer** c_buffer) noexcept:
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
    if format in ("b", "?", "h", "i", "l", "q", "n"):
        if item_size == 1:
            return item_size, NANOARROW_TYPE_INT8
        elif item_size == 2:
            return item_size, NANOARROW_TYPE_INT16
        elif item_size == 4:
            return item_size, NANOARROW_TYPE_INT32
        elif item_size == 8:
            return item_size, NANOARROW_TYPE_INT64

    # Check for unsinged integers
    if format in ("B", "H", "I", "L", "Q", "N"):
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



cdef class CDevice:
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
        return _lib_utils.device_repr(self)

    @property
    def device_type(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @staticmethod
    def resolve(ArrowDeviceType device_type, int64_t device_id):
        if device_type == ARROW_DEVICE_CPU:
            return CDEVICE_CPU
        else:
            raise ValueError(f"Device not found for type {device_type}/{device_id}")


# Cache the CPU device
# The CPU device is statically allocated (so base is None)
CDEVICE_CPU = CDevice(None, <uintptr_t>ArrowDeviceCpu())


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
        return _lib_utils.schema_repr(self)

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
    def type_id(self):
        return self._schema_view.type

    @property
    def storage_type_id(self):
        return self._schema_view.storage_type

    @property
    def type(self):
        cdef const char* type_str = ArrowTypeString(self._schema_view.type)
        if type_str != NULL:
            return type_str.decode('UTF-8')

    @property
    def storage_type(self):
        cdef const char* type_str = ArrowTypeString(self._schema_view.storage_type)
        if type_str != NULL:
            return type_str.decode('UTF-8')

    @property
    def dictionary_ordered(self):
        return self._dictionary_ordered != 0

    @property
    def nullable(self):
        return self._nullable != 0

    @property
    def map_keys_sorted(self):
        return self._map_keys_sorted != 0

    @property
    def fixed_size(self):
        if self._schema_view.type in CSchemaView._fixed_size_types:
            return self._schema_view.fixed_size

    @property
    def decimal_bitwidth(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_bitwidth

    @property
    def decimal_precision(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_precision

    @property
    def decimal_scale(self):
        if self._schema_view.type in CSchemaView._decimal_types:
            return self._schema_view.decimal_scale

    @property
    def time_unit_id(self):
        if self._schema_view.type in CSchemaView._time_unit_types:
            return self._schema_view.time_unit

    @property
    def time_unit(self):
        if self._schema_view.type in CSchemaView._time_unit_types:
            return ArrowTimeUnitString(self._schema_view.time_unit).decode('UTF-8')

    @property
    def timezone(self):
        if self._schema_view.type == NANOARROW_TYPE_TIMESTAMP:
            return self._schema_view.timezone.decode('UTF_8')

    @property
    def union_type_ids(self):
        if self._schema_view.type in CSchemaView._union_types:
            type_ids_str = self._schema_view.union_type_ids.decode('UTF-8').split(',')
            return (int(type_id) for type_id in type_ids_str)

    @property
    def extension_name(self):
        if self._schema_view.extension_name.data != NULL:
            name_bytes = PyBytes_FromStringAndSize(
                self._schema_view.extension_name.data,
                self._schema_view.extension_name.size_bytes
            )
            return name_bytes.decode('UTF-8')

    @property
    def extension_metadata(self):
        if self._schema_view.extension_name.data != NULL:
            return PyBytes_FromStringAndSize(
                self._schema_view.extension_metadata.data,
                self._schema_view.extension_metadata.size_bytes
            )


    def __repr__(self):
        return _lib_utils.schema_view_repr(self)


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

        return self

    def set_nullable(self, nullable):
        if nullable:
            self._ptr.flags = self._ptr.flags | ARROW_FLAG_NULLABLE
        else:
            self._ptr.flags = self._ptr.flags & ~ARROW_FLAG_NULLABLE

        return self

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

    @staticmethod
    def allocate(CSchema schema):
        cdef ArrowArray* c_array_out
        base = alloc_c_array(&c_array_out)
        return CArray(base, <uintptr_t>c_array_out, schema)

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowArray*>addr
        self._schema = schema

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

        # Export a shallow copy pointing to the same data in a way
        # that ensures this object stays valid.
        # TODO optimize this to export a version where children are reference
        # counted and can be released separately
        array_capsule = alloc_c_array_shallow_copy(self._base, self._ptr)
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

    @property
    def schema(self):
        return self._schema

    @property
    def length(self):
        self._assert_valid()
        return self._ptr.length

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
        return CArray(self._base, <uintptr_t>self._ptr.children[i], self._schema.child(i))

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def dictionary(self):
        self._assert_valid()
        if self._ptr.dictionary != NULL:
            return CArray(self, <uintptr_t>self._ptr.dictionary, self._schema.dictionary)
        else:
            return None

    def __repr__(self):
        return _lib_utils.array_repr(self)


cdef class CArrayView:
    """Low-level ArrowArrayView wrapper

    This object is a literal wrapper around an ArrowArrayView. It provides field accessors
    that return Python objects and handles the structure lifecycle (i.e., initialized
    ArrowArrayView structures are always released).

    See `nanoarrow.c_array_view()` for construction and usage examples.
    """
    cdef object _base
    cdef ArrowArrayView* _ptr
    cdef CDevice _device

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayView*>addr
        self._device = CDEVICE_CPU

    @property
    def storage_type(self):
        cdef const char* type_str = ArrowTypeString(self._ptr.storage_type)
        if type_str != NULL:
            return type_str.decode('UTF-8')

    @property
    def length(self):
        return self._ptr.length

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
        for i in range(3):
            if self._ptr.layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE:
                return i
        return 3

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
            self._base,
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
        return _lib_utils.array_view_repr(self)

    @staticmethod
    def from_cpu_array(CArray array):
        cdef ArrowArrayView* c_array_view
        base = alloc_c_array_view(&c_array_view)

        cdef Error error = Error()
        cdef int code = ArrowArrayViewInitFromSchema(c_array_view,
                                                       array._schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayViewInitFromSchema()", code)

        code = ArrowArrayViewSetArray(c_array_view, array._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayViewSetArray()", code)

        return CArrayView((base, array), <uintptr_t>c_array_view)


cdef class SchemaMetadata:
    """Wrapper for a lazily-parsed CSchema.metadata string
    """

    cdef object _base
    cdef const char* _metadata
    cdef ArrowMetadataReader _reader

    def __cinit__(self, object base, uintptr_t ptr):
        self._base = base
        self._metadata = <const char*>ptr

    def _init_reader(self):
        cdef int code = ArrowMetadataReaderInit(&self._reader, self._metadata)
        Error.raise_error_not_ok("ArrowMetadataReaderInit()", code)

    def __len__(self):
        self._init_reader()
        return self._reader.remaining_keys

    def __iter__(self):
        cdef ArrowStringView key
        cdef ArrowStringView value
        self._init_reader()
        while self._reader.remaining_keys > 0:
            ArrowMetadataReaderRead(&self._reader, &key, &value)
            key_obj = PyBytes_FromStringAndSize(key.data, key.size_bytes).decode('UTF-8')
            value_obj = PyBytes_FromStringAndSize(value.data, value.size_bytes)
            yield key_obj, value_obj


cdef class CBufferView:
    """Wrapper for Array buffer content

    This object is a Python wrapper around a buffer held by an Array.
    It implements the Python buffer protocol and is best accessed through
    another implementor (e.g., `np.array(array_view.buffers[1])`)). Note that
    this buffer content does not apply any parent offset.
    """
    cdef object _base
    cdef ArrowBufferView _ptr
    cdef ArrowType _buffer_data_type
    cdef CDevice _device
    cdef Py_ssize_t _element_size_bits
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides
    cdef char _format[128]

    def __cinit__(self, object base, uintptr_t addr, int64_t size_bytes,
                  ArrowType buffer_data_type,
                  Py_ssize_t element_size_bits, CDevice device):
        self._base = base
        self._ptr.data.data = <void*>addr
        self._ptr.size_bytes = size_bytes
        self._buffer_data_type = buffer_data_type
        self._device = device
        self._format[0] = 0
        self._element_size_bits = c_format_from_arrow_type(
            self._buffer_data_type,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._strides = self._item_size()
        self._shape = self._ptr.size_bytes // self._strides

    def _addr(self):
        return <uintptr_t>self._ptr.data.data

    @property
    def device_type(self):
        return self._device.device_type

    @property
    def device_id(self):
        return self._device.device_id

    @property
    def element_size_bits(self):
        return self._element_size_bits

    @property
    def size_bytes(self):
        return self._ptr.size_bytes

    @property
    def data_type_id(self):
        return self._buffer_data_type

    @property
    def data_type(self):
        return ArrowTypeString(self._buffer_data_type).decode("UTF-8")

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
        # memoryview's implementation is very fast but not always possible (half float, fixed-size binary, interval)
        if self._buffer_data_type in (
            NANOARROW_TYPE_HALF_FLOAT,
            NANOARROW_TYPE_INTERVAL_DAY_TIME,
            NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO,
            NANOARROW_TYPE_DECIMAL128,
            NANOARROW_TYPE_DECIMAL256
        ) or (
            self._buffer_data_type == NANOARROW_TYPE_BINARY and self._element_size_bits != 0
        ):
            return self._iter_struct()
        else:
            return iter(memoryview(self))

    def _iter_struct(self):
        for value in iter_unpack(self.format, self):
            if len(value) == 1:
                yield value[0]
            else:
                yield value

    cdef Py_ssize_t _item_size(self):
        if self._element_size_bits < 8:
            return 1
        else:
            return self._element_size_bits // 8

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        if self._device is not CDEVICE_CPU:
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

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __repr__(self):
        return f"CBufferView({_lib_utils.buffer_view_repr(self)})"


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
    cdef CDevice _device

    def __cinit__(self):
        self._base = None
        self._ptr = NULL
        self._data_type = NANOARROW_TYPE_BINARY
        self._element_size_bits = 0
        self._device = CDEVICE_CPU
        self._format[0] = 0

    cdef _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("CBuffer is not valid")

    def set_empty(self):
        if self._ptr == NULL:
            self._base = alloc_c_buffer(&self._ptr)
        ArrowBufferReset(self._ptr)

        self._data_type = NANOARROW_TYPE_BINARY
        self._element_size_bits = 0
        self._device = CDEVICE_CPU
        return self

    def set_pybuffer(self, obj):
        if self._ptr == NULL:
            self._base = alloc_c_buffer(&self._ptr)

        self.set_format(c_buffer_set_pybuffer(obj, &self._ptr))
        self._device = CDEVICE_CPU
        return self

    def set_format(self, str format):
        element_size_bytes, data_type = c_arrow_type_from_format(format)
        self._data_type = data_type
        self._element_size_bits = element_size_bytes * 8
        format_bytes = format.encode("UTF-8")
        snprintf(self._format, sizeof(self._format), "%s", <const char*>format_bytes)
        return self

    def set_data_type(self, ArrowType type_id, int element_size_bits=0):
        self._element_size_bits = c_format_from_arrow_type(
            type_id,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._data_type = type_id

        return self

    def _addr(self):
        self._assert_valid()
        return <uintptr_t>self._ptr.data

    @property
    def size_bytes(self):
        self._assert_valid()
        return self._ptr.size_bytes

    @property
    def capacity_bytes(self):
        self._assert_valid()
        return self._ptr.capacity_bytes

    @property
    def data(self):
        self._assert_valid()
        return CBufferView(
            self._base, <uintptr_t>self._ptr.data,
            self._ptr.size_bytes, self._data_type, self._element_size_bits,
            self._device
        )

    def __repr__(self):
        if self._ptr == NULL:
            return "CBuffer(<invalid>)"

        return _lib_utils.buffer_repr(self)


cdef class CBufferBuilder(CBuffer):
    """Wrapper around writable owned buffer CPU content"""

    def reserve_bytes(self, int64_t additional_bytes):
        self._assert_valid()
        cdef int code = ArrowBufferReserve(self._ptr, additional_bytes)
        Error.raise_error_not_ok("ArrowBufferReserve()", code)
        return self

    def write(self, content):
        self._assert_valid()

        cdef Py_buffer buffer
        cdef int64_t out
        PyObject_GetBuffer(content, &buffer, PyBUF_ANY_CONTIGUOUS)

        cdef int code = ArrowBufferReserve(self._ptr, buffer.len)
        if code != NANOARROW_OK:
            PyBuffer_Release(&buffer)
            Error.raise_error("ArrowBufferReserve()", code)

        code = PyBuffer_ToContiguous(
            self._ptr.data + self._ptr.size_bytes,
            &buffer,
            buffer.len,
            # 'C' (not sure how to pass a character literal here)
            43
        )
        out = buffer.len
        PyBuffer_Release(&buffer)
        Error.raise_error_not_ok("PyBuffer_ToContiguous()", code)

        self._ptr.size_bytes += out
        return out

    def write_values(self, obj):
        self._assert_valid()

        if self._data_type == NANOARROW_TYPE_BOOL:
            return self._write_bits(obj)

        cdef int64_t n_values = 0
        struct_obj = Struct(self._format)
        pack = struct_obj.pack
        write = self.write

        if self._data_type in (NANOARROW_TYPE_INTERVAL_DAY_TIME,
                               NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO):
            for item in obj:
                n_values += 1
                write(pack(*item))
        else:
            for item in obj:
                n_values += 1
                write(pack(item))

        return n_values

    cdef _write_bits(self, obj):
        if self._ptr.size_bytes != 0:
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
                code = ArrowBufferAppendInt8(self._ptr, buffer_item)
                Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)
                buffer_item = 0
                buffer_item_i = 0

        if buffer_item_i != 0:
            code = ArrowBufferAppendInt8(self._ptr, buffer_item)
            Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)

        return n_values

    def finish(self):
        return self


cdef class CArrayBuilder:
    cdef CArray c_array
    cdef ArrowArray* _ptr

    def __cinit__(self, CArray array):
        self.c_array = array
        self._ptr = array._ptr

    @staticmethod
    def allocate():
        return CArrayBuilder(CArray.allocate(CSchema.allocate()))

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

    def set_buffer(self, int64_t i, CBuffer buffer):
        self.c_array._assert_valid()
        cdef int code = ArrowArraySetBuffer(self._ptr, i, buffer._ptr)
        Error.raise_error_not_ok("ArrowArraySetBuffer()", code)
        return self

    def set_child(self, int64_t i, CArray c_array):
        cdef CArray child = self.c_array.child(i)
        if child._ptr.release != NULL:
            ArrowArrayRelease(child._ptr)
        ArrowArrayMove(c_array._ptr, child._ptr)
        return self

    def finish(self, validation_level="default"):
        self.c_array._assert_valid()
        cdef ArrowValidationLevel c_validation_level = NANOARROW_VALIDATION_LEVEL_DEFAULT
        if validation_level == "full":
            c_validation_level = NANOARROW_VALIDATION_LEVEL_FULL
        elif validation_level == "minimal":
            c_validation_level = NANOARROW_VALIDATION_LEVEL_MINIMAL
        elif validation_level == "none":
            c_validation_level = NANOARROW_VALIDATION_LEVEL_NONE

        cdef Error error = Error()
        cdef int code = ArrowArrayFinishBuilding(self._ptr, c_validation_level, &error.c_error)
        error.raise_message_not_ok("ArrowArrayFinishBuildingDefault()", code)

        return self.c_array


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

    @staticmethod
    def allocate():
        cdef ArrowArrayStream* c_array_stream_out
        base = alloc_c_array_stream(&c_array_stream_out)
        return CArrayStream(base, <uintptr_t>c_array_stream_out)

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayStream*>addr
        self._cached_schema = None

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
        cdef int code = self._ptr.get_schema(self._ptr, schema._ptr)
        Error.raise_error_not_ok("ArrowArrayStream::get_schema()", code)

        self._cached_schema = schema

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
        if self._cached_schema is None:
            self._cached_schema = CSchema.allocate()
            self._get_schema(self._cached_schema)

        cdef Error error = Error()
        cdef CArray array = CArray.allocate(self._cached_schema)
        cdef int code = ArrowArrayStreamGetNext(self._ptr, array._ptr, &error.c_error)
        Error.raise_error_not_ok("ArrowArrayStream::get_next()", code)

        if not array.is_valid():
            raise StopIteration()
        else:
            return array

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    def __repr__(self):
        return _lib_utils.array_stream_repr(self)


cdef class CDeviceArray:
    cdef object _base
    cdef ArrowDeviceArray* _ptr
    cdef CSchema _schema

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowDeviceArray*>addr
        self._schema = schema

    @property
    def device_type(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @property
    def array(self):
        return CArray(self, <uintptr_t>&self._ptr.array, self._schema)

    def __repr__(self):
        return _lib_utils.device_array_repr(self)
