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
be literal and stay close to the structure definitions.
"""

from libc.stdint cimport uintptr_t, int64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython cimport Py_buffer
from nanoarrow_c cimport *
from nanoarrow_device_c cimport *

def c_version():
    """Return the nanoarrow C library version string
    """
    return ArrowNanoarrowVersion().decode("UTF-8")


cdef class SchemaHolder:
    """Memory holder for an ArrowSchema

    This class is responsible for the lifecycle of the ArrowSchema
    whose memory it is responsible for. When this object is deleted,
    a non-NULL release callback is invoked.
    """
    cdef ArrowSchema c_schema

    def __cinit__(self):
        self.c_schema.release = NULL

    def __dealloc__(self):
        if self.c_schema.release != NULL:
          self.c_schema.release(&self.c_schema)

    def _addr(self):
        return <uintptr_t>&self.c_schema


cdef class ArrayHolder:
    """Memory holder for an ArrowArray

    This class is responsible for the lifecycle of the ArrowArray
    whose memory it is responsible. When this object is deleted,
    a non-NULL release callback is invoked.
    """
    cdef ArrowArray c_array

    def __cinit__(self):
        self.c_array.release = NULL

    def __dealloc__(self):
        if self.c_array.release != NULL:
          self.c_array.release(&self.c_array)

    def _addr(self):
        return <uintptr_t>&self.c_array

cdef class ArrayStreamHolder:
    """Memory holder for an ArrowArrayStream

    This class is responsible for the lifecycle of the ArrowArrayStream
    whose memory it is responsible. When this object is deleted,
    a non-NULL release callback is invoked.
    """
    cdef ArrowArrayStream c_array_stream

    def __cinit__(self):
        self.c_array_stream.release = NULL

    def __dealloc__(self):
        if self.c_array_stream.release != NULL:
          self.c_array_stream.release(&self.c_array_stream)

    def _addr(self):
        return <uintptr_t>&self.c_array_stream


cdef class ArrayViewHolder:
    """Memory holder for an ArrowArrayView

    This class is responsible for the lifecycle of the ArrowArrayView
    whose memory it is responsible. When this object is deleted,
    ArrowArrayViewReset() is called on the contents.
    """
    cdef ArrowArrayView c_array_view

    def __cinit__(self):
        ArrowArrayViewInitFromType(&self.c_array_view, NANOARROW_TYPE_UNINITIALIZED)

    def __dealloc__(self):
        ArrowArrayViewReset(&self.c_array_view)

    def _addr(self):
        return <uintptr_t>&self.c_array_view


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

    @staticmethod
    def raise_error(what, code):
        """Raise a NanoarrowException without a message
        """
        raise NanoarrowException(what, code, "")


cdef class Schema:
    """ArrowSchema wrapper

    This class provides a user-facing interface to access the fields of
    an ArrowSchema as defined in the Arrow C Data interface. These objects
    are usually created using `nanoarrow.schema()`. This Python wrapper
    allows access to schema fields but does not automatically deserialize
    their content: use `.view()` to validate and deserialize the content
    into a more easily inspectable object.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.schema(pa.int32())
    >>> schema.is_valid()
    True
    >>> schema.format
    'i'
    >>> schema.name
    ''
    >>> schema_view = schema.view()
    >>> schema_view.type
    'int32'
    """
    cdef object _base
    cdef ArrowSchema* _ptr

    @staticmethod
    def allocate():
        base = SchemaHolder()
        return Schema(base, base._addr())

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowSchema*>addr

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
        return Schema(
            schema_capsule,
            <uintptr_t>PyCapsule_GetPointer(schema_capsule, 'arrow_schema')
        )

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("schema is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("schema is released")

    def __repr__(self):
        cdef int64_t n_chars = ArrowSchemaToString(self._ptr, NULL, 0, True)
        cdef char* out = <char*>PyMem_Malloc(n_chars + 1)
        if not out:
            raise MemoryError()

        ArrowSchemaToString(self._ptr, out, n_chars + 1, True)
        out_str = out.decode("UTF-8")
        PyMem_Free(out)

        return out_str

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
            return SchemaMetadata(self, <uintptr_t>self._ptr.metadata)
        else:
            return None

    @property
    def children(self):
        self._assert_valid()
        return SchemaChildren(self)

    @property
    def dictionary(self):
        self._assert_valid()
        if self._ptr.dictionary != NULL:
            return Schema(self, <uintptr_t>self._ptr.dictionary)
        else:
            return None

    def view(self):
        self._assert_valid()
        schema_view = SchemaView()
        cdef Error error = Error()
        cdef int result = ArrowSchemaViewInit(&schema_view._schema_view, self._ptr, &error.c_error)
        if result != NANOARROW_OK:
            error.raise_message("ArrowSchemaViewInit()", result)

        return schema_view


cdef class SchemaView:
    """ArrowSchemaView wrapper

    The ArrowSchemaView is a nanoarrow C library structure that facilitates
    access to the deserialized content of an ArrowSchema (e.g., parameter
    values for parameterized types). This wrapper extends that facility to Python.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.schema(pa.decimal128(10, 3))
    >>> schema_view = schema.view()
    >>> schema_view.type
    'decimal128'
    >>> schema_view.decimal_bitwidth
    128
    >>> schema_view.decimal_precision
    10
    >>> schema_view.decimal_scale
    3
    """
    cdef ArrowSchemaView _schema_view

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

    def __cinit__(self):
        self._schema_view.type = NANOARROW_TYPE_UNINITIALIZED
        self._schema_view.storage_type = NANOARROW_TYPE_UNINITIALIZED

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
    def fixed_size(self):
        if self._schema_view.type in SchemaView._fixed_size_types:
            return self._schema_view.fixed_size

    @property
    def decimal_bitwidth(self):
        if self._schema_view.type in SchemaView._decimal_types:
            return self._schema_view.decimal_bitwidth

    @property
    def decimal_precision(self):
        if self._schema_view.type in SchemaView._decimal_types:
            return self._schema_view.decimal_precision

    @property
    def decimal_scale(self):
        if self._schema_view.type in SchemaView._decimal_types:
            return self._schema_view.decimal_scale

    @property
    def time_unit(self):
        if self._schema_view.type in SchemaView._time_unit_types:
            return ArrowTimeUnitString(self._schema_view.time_unit).decode('UTF-8')

    @property
    def timezone(self):
        if self._schema_view.type == NANOARROW_TYPE_TIMESTAMP:
            return self._schema_view.timezone.decode('UTF_8')

    @property
    def union_type_ids(self):
        if self._schema_view.type in SchemaView._union_types:
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

cdef class Array:
    """ArrowArray wrapper

    This class provides a user-facing interface to access the fields of
    an ArrowArray as defined in the Arrow C Data interface, holding an
    optional reference to a Schema that can be used to safely deserialize
    the content. These objects are usually created using `nanoarrow.array()`.
    This Python wrapper allows access to array fields but does not
    automatically deserialize their content: use `.view()` to validate and
    deserialize the content into a more easily inspectable object.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import numpy as np
    >>> import nanoarrow as na
    >>> array = na.array(pa.array(["one", "two", "three", None]))
    >>> array.length
    4
    >>> array.null_count
    1
    >>> array_view = array.view()
    """
    cdef object _base
    cdef ArrowArray* _ptr
    cdef Schema _schema

    @staticmethod
    def allocate(Schema schema):
        base = ArrayHolder()
        return Array(base, base._addr(), schema)

    def __cinit__(self, object base, uintptr_t addr, Schema schema):
        self._base = base,
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
            Schema out_schema
            Array out

        out_schema = Schema._import_from_c_capsule(schema_capsule)
        out = Array(
            array_capsule,
            <uintptr_t>PyCapsule_GetPointer(array_capsule, 'arrow_array'),
            out_schema
        )

        return out

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("Array is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("Array is released")

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
        return self._ptr.null_count

    @property
    def buffers(self):
        return tuple(<uintptr_t>self._ptr.buffers[i] for i in range(self._ptr.n_buffers))

    @property
    def children(self):
        return ArrayChildren(self)

    @property
    def dictionary(self):
        self._assert_valid()
        if self._ptr.dictionary != NULL:
            return Array(self, <uintptr_t>self._ptr.dictionary, self._schema.dictionary)
        else:
            return None

    def view(self):
        cdef ArrayViewHolder holder = ArrayViewHolder()

        cdef Error error = Error()
        cdef int result = ArrowArrayViewInitFromSchema(&holder.c_array_view,
                                                       self._schema._ptr, &error.c_error)
        if result != NANOARROW_OK:
            error.raise_message("ArrowArrayViewInitFromSchema()", result)

        result = ArrowArrayViewSetArray(&holder.c_array_view, self._ptr, &error.c_error)
        if result != NANOARROW_OK:
            error.raise_message("ArrowArrayViewSetArray()", result)

        return ArrayView(holder, holder._addr(), self._schema, self)


cdef class ArrayView:
    """ArrowArrayView wrapper

    The ArrowArrayView is a nanoarrow C library structure that provides
    structured access to buffers addresses, buffer sizes, and buffer
    data types. The buffer data is usually propagated from an ArrowArray
    but can also be propagated from other types of objects (e.g., serialized
    IPC). The offset and length of this view are independent of its parent
    (i.e., this object can also represent a slice of its parent).

    Examples
    --------

    >>> import pyarrow as pa
    >>> import numpy as np
    >>> import nanoarrow as na
    >>> array_view = na.array(pa.array(["one", "two", "three", None])).view()
    >>> np.array(array_view.buffers[1])
    array([ 0,  3,  6, 11, 11], dtype=int32)
    >>> np.array(array_view.buffers[2])
    array([b'o', b'n', b'e', b't', b'w', b'o', b't', b'h', b'r', b'e', b'e'],
          dtype='|S1')
    """
    cdef object _base
    cdef ArrowArrayView* _ptr
    cdef Schema _schema
    cdef object _base_buffer

    def __cinit__(self, object base, uintptr_t addr, Schema schema, object base_buffer):
        self._base = base
        self._ptr = <ArrowArrayView*>addr
        self._schema = schema
        self._base_buffer = base_buffer

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
    def children(self):
        return ArrayViewChildren(self)

    @property
    def buffers(self):
        return ArrayViewBuffers(self)

    @property
    def dictionary(self):
        if self._ptr.dictionary == NULL:
            return None
        else:
            return ArrayView(
                self,
                <uintptr_t>self._ptr.dictionary,
                self._schema.dictionary,
                None
            )

    @property
    def schema(self):
        return self._schema


cdef class SchemaChildren:
    """Wrapper for a lazily-resolved list of Schema children
    """
    cdef Schema _parent
    cdef int64_t _length

    def __cinit__(self, Schema parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")

        return Schema(self._parent, self._child_addr(k))

    cdef _child_addr(self, int64_t i):
        cdef ArrowSchema** children = self._parent._ptr.children
        cdef ArrowSchema* child = children[i]
        return <uintptr_t>child


cdef class SchemaMetadata:
    """Wrapper for a lazily-parsed Schema.metadata string
    """

    cdef object _parent
    cdef const char* _metadata
    cdef ArrowMetadataReader _reader

    def __cinit__(self, object parent, uintptr_t ptr):
        self._parent = parent
        self._metadata = <const char*>ptr

    def _init_reader(self):
        cdef int result = ArrowMetadataReaderInit(&self._reader, self._metadata)
        if result != NANOARROW_OK:
            Error.raise_error("ArrowMetadataReaderInit()", result)

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


cdef class ArrayChildren:
    """Wrapper for a lazily-resolved list of Array children
    """
    cdef Array _parent
    cdef int64_t _length

    def __cinit__(self, Array parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")
        return Array(self._parent, self._child_addr(k), self._parent.schema.children[k])

    cdef _child_addr(self, int64_t i):
        cdef ArrowArray** children = self._parent._ptr.children
        cdef ArrowArray* child = children[i]
        return <uintptr_t>child


cdef class ArrayViewChildren:
    """Wrapper for a lazily-resolved list of ArrayView children
    """
    cdef ArrayView _parent
    cdef int64_t _length

    def __cinit__(self, ArrayView parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")
        return ArrayView(
            self._parent,
            self._child_addr(k),
            self._parent._schema.children[k],
            None
        )

    cdef _child_addr(self, int64_t i):
        cdef ArrowArrayView** children = self._parent._ptr.children
        cdef ArrowArrayView* child = children[i]
        return <uintptr_t>child


cdef class BufferView:
    """Wrapper for Array buffer content

    This object is a Python wrapper around a buffer held by an Array.
    It implements the Python buffer protocol and is best accessed through
    another implementor (e.g., `np.array(array_view.buffers[1])`)). Note that
    this buffer content does not apply any parent offset.
    """
    cdef object _base
    cdef ArrowBufferView* _ptr
    cdef ArrowBufferType _buffer_type
    cdef ArrowType _buffer_data_type
    cdef Py_ssize_t _element_size_bits
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides

    def __cinit__(self, object base, uintptr_t addr,
                 ArrowBufferType buffer_type, ArrowType buffer_data_type,
                 Py_ssize_t element_size_bits):
        self._base = base
        self._ptr = <ArrowBufferView*>addr
        self._buffer_type = buffer_type
        self._buffer_data_type = buffer_data_type
        self._element_size_bits = element_size_bits
        self._strides = self._item_size()
        self._shape = self._ptr.size_bytes // self._strides


    cdef Py_ssize_t _item_size(self):
        if self._buffer_data_type == NANOARROW_TYPE_BOOL:
            return 1
        elif self._buffer_data_type == NANOARROW_TYPE_STRING:
            return 1
        elif self._buffer_data_type == NANOARROW_TYPE_BINARY:
            return 1
        else:
            return self._element_size_bits // 8

    cdef const char* _get_format(self):
        if self._buffer_data_type == NANOARROW_TYPE_INT8:
            return "b"
        elif self._buffer_data_type == NANOARROW_TYPE_UINT8:
            return "B"
        elif self._buffer_data_type == NANOARROW_TYPE_INT16:
            return "h"
        elif self._buffer_data_type == NANOARROW_TYPE_UINT16:
            return "H"
        elif self._buffer_data_type == NANOARROW_TYPE_INT32:
            return "i"
        elif self._buffer_data_type == NANOARROW_TYPE_UINT32:
            return "I"
        elif self._buffer_data_type == NANOARROW_TYPE_INT64:
            return "l"
        elif self._buffer_data_type == NANOARROW_TYPE_UINT64:
            return "L"
        elif self._buffer_data_type == NANOARROW_TYPE_FLOAT:
            return "f"
        elif self._buffer_data_type == NANOARROW_TYPE_DOUBLE:
            return "d"
        elif self._buffer_data_type == NANOARROW_TYPE_STRING:
            return "c"
        else:
            return "B"

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = <void*>self._ptr.data.data
        buffer.format = self._get_format()
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


cdef class ArrayViewBuffers:
    """A lazily-resolved list of ArrayView buffers
    """
    cdef ArrayView _array_view
    cdef int64_t _length

    def __cinit__(self, ArrayView array_view):
        self._array_view = array_view
        self._length = 3
        for i in range(3):
            if self._array_view._ptr.layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE:
                self._length = i
                break

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")
        cdef ArrowBufferView* buffer_view = &(self._array_view._ptr.buffer_views[k])
        if buffer_view.data.data == NULL:
            return None

        return BufferView(
            self._array_view,
            <uintptr_t>buffer_view,
            self._array_view._ptr.layout.buffer_type[k],
            self._array_view._ptr.layout.buffer_data_type[k],
            self._array_view._ptr.layout.element_size_bits[k]
        )


cdef class ArrayStream:
    """ArrowArrayStream wrapper

    This class provides a user-facing interface to access the fields of
    an ArrowArrayStream as defined in the Arrow C Stream interface.
    These objects are usually created using `nanoarrow.array_stream()`.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> pa_column = pa.array([1, 2, 3], pa.int32())
    >>> pa_batch = pa.record_batch([pa_column], names=["col1"])
    >>> pa_reader = pa.RecordBatchReader.from_batches(pa_batch.schema, [pa_batch])
    >>> array_stream = na.array_stream(pa_reader)
    >>> array_stream.get_schema()
    struct<col1: int32>
    >>> array_stream.get_next().length
    3
    >>> array_stream.get_next() is None
    Traceback (most recent call last):
      ...
    StopIteration
    """
    cdef object _base
    cdef ArrowArrayStream* _ptr
    cdef object _cached_schema

    @staticmethod
    def allocate():
        base = ArrayStreamHolder()
        return ArrayStream(base, base._addr())

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
        return ArrayStream(
            stream_capsule,
            <uintptr_t>PyCapsule_GetPointer(stream_capsule, 'arrow_array_stream')
        )

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("array stream pointer is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("array stream is released")

    def _get_schema(self, Schema schema):
        self._assert_valid()
        cdef int code = self._ptr.get_schema(self._ptr, schema._ptr)
        cdef const char* message = NULL
        if code != NANOARROW_OK:
            message = self._ptr.get_last_error(self._ptr)
            if message != NULL:
                raise NanoarrowException(
                    "ArrowArrayStream::get_schema()",
                    code,
                    message.decode("UTF-8")
                )
            else:
                raise NanoarrowException("ArrowArrayStream::get_schema()", code)

        self._cached_schema = schema

    def get_schema(self):
        """Get the schema associated with this stream
        """
        out = Schema.allocate()
        self._get_schema(out)
        return out

    def get_next(self):
        """Get the next Array from this stream

        Returns None when there are no more arrays in this stream.
        """
        self._assert_valid()

        # We return a reference to the same Python object for each
        # Array that is returned. This is independent of get_schema(),
        # which is guaranteed to call the C object's callback and
        # faithfully pass on the returned value.
        if self._cached_schema is None:
            self._cached_schema = Schema.allocate()
            self._get_schema(self._cached_schema)

        cdef Array array = Array.allocate(self._cached_schema)
        cdef int code = self._ptr.get_next(self._ptr, array._ptr)
        cdef const char* message = NULL
        if code != NANOARROW_OK:
            message = self._ptr.get_last_error(self._ptr)
            if message != NULL:
                raise NanoarrowException(
                    "ArrowArrayStream::get_next()",
                    code,
                    message.decode("UTF-8")
                )
            else:
                raise NanoarrowException("ArrowArrayStream::get_next()", code)

        if not array.is_valid():
            raise StopIteration()
        else:
            return array

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    @staticmethod
    def allocate():
        base = ArrayStreamHolder()
        return ArrayStream(base, base._addr())


cdef class DeviceArrayHolder:
    """Memory holder for an ArrowDeviceArray

    This class is responsible for the lifecycle of the ArrowDeviceArray
    whose memory it is responsible. When this object is deleted,
    a non-NULL release callback is invoked.
    """
    cdef ArrowDeviceArray c_array

    def __cinit__(self):
        self.c_array.array.release = NULL

    def __dealloc__(self):
        if self.c_array.array.release != NULL:
          self.c_array.array.release(&self.c_array.array)

    def _addr(self):
        return <uintptr_t>&self.c_array

cdef class DeviceHolder:
    """Memory holder for an ArrowDevice

    This class is responsible for the lifecycle of the ArrowDevice
    whose memory it is responsible. When this object is deleted,
    a non-NULL release callback is invoked.
    """

    cdef ArrowDevice c_device

    def __cinit__(self):
        self.c_device.release = NULL

    def __dealloc__(self):
        if self.c_device.release != NULL:
          self.c_device.release(&self.c_device)

    def _addr(self):
        return <uintptr_t>&self.c_device


cdef class Device:
    cdef object _base
    cdef ArrowDevice* _ptr

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowDevice*>addr

    def array_init(self, uintptr_t array_addr, Schema schema):
        cdef ArrowArray* array_ptr = <ArrowArray*>array_addr
        cdef DeviceArrayHolder holder = DeviceArrayHolder()
        cdef int result = ArrowDeviceArrayInit(self._ptr, &holder.c_array, array_ptr)
        if result != NANOARROW_OK:
            Error.raise_error("ArrowDevice::init_array", result)

        return DeviceArray(holder, holder._addr(), schema)

    @property
    def device_type(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @staticmethod
    def resolve(ArrowDeviceType device_type, int64_t device_id):
        if device_type == ARROW_DEVICE_CPU:
            return Device.cpu()
        else:
            raise ValueError(f"Device not found for type {device_type}/{device_id}")

    @staticmethod
    def cpu():
        return Device(None, <uintptr_t>ArrowDeviceCpu())


cdef class DeviceArray:
    cdef object _base
    cdef ArrowDeviceArray* _ptr
    cdef Schema _schema

    def __cinit__(self, object base, uintptr_t addr, Schema schema):
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
        return Array(self, <uintptr_t>&self._ptr.array, self._schema)
