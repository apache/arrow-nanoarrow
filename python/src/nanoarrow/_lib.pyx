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

"""Low-level nanoarrow Python bindings."""

from libc.stdint cimport uintptr_t, int64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython cimport Py_buffer
from nanoarrow_c cimport *

def c_version():
    return ArrowNanoarrowVersion().decode("UTF-8")

cdef class SchemaHolder:
    cdef ArrowSchema c_schema

    def __cinit__(self):
        self.c_schema.release = NULL

    def __dealloc__(self):
        if self.c_schema.release != NULL:
          self.c_schema.release(&self.c_schema)

    def _addr(self):
        return <uintptr_t>&self.c_schema

cdef class ArrayHolder:
    cdef ArrowArray c_array

    def __cinit__(self):
        self.c_array.release = NULL

    def __dealloc__(self):
        if self.c_array.release != NULL:
          self.c_array.release(&self.c_array)

    def _addr(self):
        return <uintptr_t>&self.c_array

cdef class ArrayStreamHolder:
    cdef ArrowArrayStream c_array_stream

    def __cinit__(self):
        self.c_array_stream.release = NULL

    def __dealloc__(self):
        if self.c_array_stream.release != NULL:
          self.c_array_stream.release(&self.c_array_stream)

    def _addr(self):
        return <uintptr_t>&self.c_array_stream

cdef class ArrayViewHolder:
    cdef ArrowArrayView c_array_view

    def __init__(self):
        ArrowArrayViewInitFromType(&self.c_array_view, NANOARROW_TYPE_UNINITIALIZED)

    def __dealloc__(self):
        ArrowArrayViewReset(&self.c_array_view)

    def _addr(self):
        return <uintptr_t>&self.c_array_view


class NanoarrowException(RuntimeError):

    def __init__(self, what, code, message):
        self.what = what
        self.code = code
        self.message = message

        if self.message == "":
            super().__init__(f"{self.what} failed ({self.code})")
        else:
            super().__init__(f"{self.what} failed ({self.code}): {self.message}")


cdef class Error:
    cdef ArrowError c_error

    def __cinit__(self):
        self.c_error.message[0] = 0

    def raise_message(self, what, code):
        raise Exception(what, code, self.c_error.message.decode("UTF-8"))

    @staticmethod
    def raise_error(what, code):
        raise Exception(what, code, "")


cdef class Schema:
    cdef object _base
    cdef ArrowSchema* _ptr

    @staticmethod
    def empty():
        base = SchemaHolder()
        return Schema(base, base._addr())

    def __init__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowSchema*>addr

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr.release != NULL

    def _assert_valid(self):
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
        cdef ArrowError error
        cdef int result = ArrowSchemaViewInit(&schema_view._schema_view, self._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))
        return schema_view

cdef class SchemaView:
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

    def __init__(self):
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
    cdef object _base
    cdef ArrowArray* _ptr
    cdef Schema _schema

    @staticmethod
    def empty(Schema schema):
        base = ArrayHolder()
        return Array(base, base._addr(), schema)

    def __init__(self, object base, uintptr_t addr, Schema schema):
        self._base = base,
        self._ptr = <ArrowArray*>addr
        self._schema = schema

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr.release != NULL

    def _assert_valid(self):
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

        cdef ArrowError error
        cdef int result = ArrowArrayViewInitFromSchema(&holder.c_array_view,
                                                       self._schema._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))

        result = ArrowArrayViewSetArray(&holder.c_array_view, self._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))

        return ArrayView(holder, holder._addr(), self)


cdef class ArrayView:
    cdef object _base
    cdef ArrowArrayView* _ptr
    cdef Array _array

    def __init__(self, object base, uintptr_t addr, Array array):
        self._base = base,
        self._ptr = <ArrowArrayView*>addr
        self._array = array

    @property
    def children(self):
        return ArrayViewChildren(self)

    @property
    def buffers(self):
        return ArrayViewBuffers(self)

    @property
    def dictionary(self):
        return ArrayView(self, <uintptr_t>self._ptr.dictionary, self._array.dictionary)

    @property
    def array(self):
        return self._array

    @property
    def schema(self):
        return self._array._schema

cdef class SchemaChildren:
    cdef Schema _parent
    cdef int64_t _length

    def __init__(self, Schema parent):
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
    cdef object _parent
    cdef const char* _metadata
    cdef ArrowMetadataReader _reader

    def __init__(self, object parent, uintptr_t ptr):
        self._parent = parent
        self._metadata = <const char*>ptr

    def _init_reader(self):
        cdef int result = ArrowMetadataReaderInit(&self._reader, self._metadata)
        if result != NANOARROW_OK:
            raise ValueError('ArrowMetadataReaderInit() failed')

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
    cdef Array _parent
    cdef int64_t _length

    def __init__(self, Array parent):
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
    cdef ArrayView _parent
    cdef int64_t _length

    def __init__(self, ArrayView parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")
        return ArrayView(self._parent, self._child_addr(k), self._parent._array.children[k])

    cdef _child_addr(self, int64_t i):
        cdef ArrowArrayView** children = self._parent._ptr.children
        cdef ArrowArrayView* child = children[i]
        return <uintptr_t>child

cdef class BufferView:
    cdef object _base
    cdef ArrowBufferView* _ptr
    cdef ArrowBufferType _buffer_type
    cdef ArrowType _buffer_data_type
    cdef Py_ssize_t _element_size_bits
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides

    def __init__(self, object base, uintptr_t addr,
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
        buffer.buf = self._ptr.data.data
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
    cdef ArrayView _array_view
    cdef int64_t _length

    def __init__(self, ArrayView array_view):
        self._array_view = array_view
        self._length = array_view._array._ptr.n_buffers

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
    cdef object _base
    cdef ArrowArrayStream* _ptr

    def __init__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowArrayStream*>addr
        self._cached_schema = None

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
                Error.raise_error("ArrowArrayStream::get_schema()", code)

        self._cached_schema = schema

    def get_schema(self):
        # Update the cached copy of the schema as an independent object
        if self._cached_schema is not None:
            del self._cached_schema
        self._cached_schema = Schema.empty()
        self._get_schema(self._cached_schema)

        # Return an independent copy
        out = Schema.empty()
        self._get_schema(out)
        return out

    def get_next(self):
        self._assert_valid()

        if self._cached_schema is None:
            self._cached_schema = Schema.empty()
            self._get_schema(self._cached_schema)

        cdef Array array = Array.empty(self._cached_schema)
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
                Error.raise_error("ArrowArrayStream::get_next()", code)

        return array
