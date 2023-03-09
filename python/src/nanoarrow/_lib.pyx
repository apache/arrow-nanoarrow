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

from libc.stdint cimport uint8_t, uintptr_t, int64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython cimport Py_buffer
from nanoarrow_c cimport *

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef dict _numpy_type_map = {
    NANOARROW_TYPE_UINT8: cnp.NPY_UINT8,
    NANOARROW_TYPE_INT8: cnp.NPY_INT8,
    NANOARROW_TYPE_UINT16: cnp.NPY_UINT16,
    NANOARROW_TYPE_INT16: cnp.NPY_INT16,
    NANOARROW_TYPE_UINT32: cnp.NPY_UINT32,
    NANOARROW_TYPE_INT32: cnp.NPY_INT32,
    NANOARROW_TYPE_UINT64: cnp.NPY_UINT64,
    NANOARROW_TYPE_INT64: cnp.NPY_INT64,
    NANOARROW_TYPE_HALF_FLOAT: cnp.NPY_FLOAT16,
    NANOARROW_TYPE_FLOAT: cnp.NPY_FLOAT32,
    NANOARROW_TYPE_DOUBLE: cnp.NPY_FLOAT64,
}


def as_numpy_array(arr):
    cdef ArrowSchema schema
    cdef ArrowArray array
    cdef ArrowArrayView array_view
    cdef ArrowError error

    arr._export_to_c(<uintptr_t> &array, <uintptr_t> &schema)
    ArrowArrayViewInitFromSchema(&array_view, &schema, &error)

    # primitive arrays have DATA as the second buffer
    if array_view.layout.buffer_type[1] != NANOARROW_BUFFER_TYPE_DATA:
        raise TypeError("Cannot convert a non-primitive array")

    # disallow nulls for this method
    if array.null_count > 0:
        raise ValueError("Cannot convert array with nulls")
    elif array.null_count < 0:
        # not yet computed
        if array_view.layout.buffer_type[0] == NANOARROW_BUFFER_TYPE_VALIDITY:
            if array.buffers[0] != NULL:
                null_count = ArrowBitCountSet(
                    <const uint8_t *>array.buffers[0], array.offset, array.length
                )
                if null_count > 0:
                    raise ValueError("Cannot convert array with nulls")

    cdef int type_num
    if array_view.storage_type in _numpy_type_map:
        type_num = _numpy_type_map[array_view.storage_type]
    else:
        raise NotImplementedError(array_view.storage_type)

    cdef cnp.npy_intp dims[1]
    dims[0] = array.length
    cdef cnp.ndarray result = cnp.PyArray_New(
        np.ndarray, 1, dims, type_num, NULL, <void *> array.buffers[1], -1, 0, <object>NULL
    )
    # TODO set base

    return result


def version():
    return ArrowNanoarrowVersion().decode("UTF-8")

cdef class CSchemaHolder:
    cdef ArrowSchema c_schema

    def __init__(self):
        self.c_schema.release = NULL

    def __del__(self):
        if self.c_schema.release != NULL:
          self.c_schema.release(&self.c_schema)

    def _addr(self):
        return <uintptr_t>&self.c_schema

cdef class CArrayHolder:
    cdef ArrowArray c_array

    def __init__(self):
        self.c_array.release = NULL

    def __del__(self):
        if self.c_array.release != NULL:
          self.c_array.release(&self.c_array)

    def _addr(self):
        return <uintptr_t>&self.c_array

cdef class CArrayViewHolder:
    cdef ArrowArrayView c_array_view

    def __init__(self):
        ArrowArrayViewInitFromType(&self.c_array_view, NANOARROW_TYPE_UNINITIALIZED)

    def __del__(self):
        ArrowArrayViewReset(&self.c_array_view)

    def _addr(self):
        return <uintptr_t>&self.c_array_view

cdef class CSchema:
    cdef object _base
    cdef ArrowSchema* _ptr

    @staticmethod
    def Empty():
        base = CSchemaHolder()
        return CSchema(base, base._addr())

    def __init__(self, object base, uintptr_t addr):
        self._base = base,
        self._ptr = <ArrowSchema*>addr

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr.release != NULL

    cdef void _assert_valid(self):
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
    def children(self):
        self._assert_valid()
        return CSchemaChildren(self)

    def parse(self):
        self._assert_valid()

        cdef ArrowError error
        cdef ArrowSchemaView schema_view

        cdef int result = ArrowSchemaViewInit(&schema_view, self._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))

        out = {
            'name': self._ptr.name.decode('UTF-8') if self._ptr.name else None,
            'type': ArrowTypeString(schema_view.type).decode('UTF-8'),
            'storage_type': ArrowTypeString(schema_view.storage_type).decode('UTF-8')
        }

        if schema_view.storage_type in (NANOARROW_TYPE_FIXED_SIZE_LIST,
                                        NANOARROW_TYPE_FIXED_SIZE_BINARY):
            out['fixed_size'] = schema_view.fixed_size

        if schema_view.storage_type in (NANOARROW_TYPE_DECIMAL128,
                                        NANOARROW_TYPE_DECIMAL256):
            out['decimal_bitwidth'] = schema_view.decimal_bitwidth
            out['decimal_precision'] = schema_view.decimal_precision
            out['decimal_scale'] = schema_view.decimal_scale

        return out

cdef class CArray:
    cdef object _base
    cdef ArrowArray* _ptr
    cdef CSchema _schema

    @staticmethod
    def Empty(CSchema schema):
        base = CArrayHolder()
        return CArray(base, base._addr(), schema)

    def __init__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base,
        self._ptr = <ArrowArray*>addr
        self._schema = schema

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr.release != NULL

    cdef void _assert_valid(self):
        if self._ptr.release == NULL:
            raise RuntimeError("Array is released")

    @property
    def schema(self):
        return self._schema

    @property
    def children(self):
        return CArrayChildren(self)

    def validate(self):
        cdef CArrayViewHolder holder = CArrayViewHolder()

        cdef ArrowError error
        cdef int result = ArrowArrayViewInitFromSchema(&holder.c_array_view,
                                                       self._schema._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))

        result = ArrowArrayViewSetArray(&holder.c_array_view, self._ptr, &error)
        if result != NANOARROW_OK:
            raise ValueError(ArrowErrorMessage(&error))

        return CArrayView(holder, holder._addr(), self)


cdef class CBufferView:
    cdef object _base
    cdef ArrowBufferView* _ptr
    cdef Py_ssize_t _shape
    cdef Py_ssize_t _strides

    def __init__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowBufferView*>addr
        self._shape = self._ptr.size_bytes
        self._strides = 1

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = self._ptr.data.data
        buffer.format = NULL
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = self._ptr.size_bytes
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = &self._shape
        buffer.strides = &self._strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class CArrayViewBuffers:
    cdef CArrayView _array_view
    cdef int64_t _length

    def __init__(self, CArrayView array_view):
        self._array_view = array_view
        self._length = array_view._array._ptr.n_buffers

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")
        cdef ArrowBufferView* buffer_view = &(self._array_view._ptr.buffer_views[k])
        return CBufferView(self._array_view, <uintptr_t>buffer_view)

cdef class CArrayView:
    cdef object _base
    cdef ArrowArrayView* _ptr
    cdef CArray _array

    def __init__(self, object base, uintptr_t addr, CArray array):
        self._base = base,
        self._ptr = <ArrowArrayView*>addr
        self._array = array

    @property
    def children(self):
        return CArrayViewChildren(self)

    @property
    def buffers(self):
        return CArrayViewBuffers(self)

    @property
    def array(self):
        return self._array

    @property
    def schema(self):
        return self._array._schema

    def __len__(self):
        return self._ptr.array.length

    def value_int(self, int64_t i):
        if i < 0 or i >= self._ptr.array.length:
            raise IndexError()
        return ArrowArrayViewGetIntUnsafe(self._ptr, i)

cdef class CSchemaChildren:
    cdef CSchema _parent
    cdef int64_t _length

    def __init__(self, CSchema parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")

        return CSchema(self._parent, self._child_addr(k))

    cdef _child_addr(self, int64_t i):
        cdef ArrowSchema** children = self._parent._ptr.children
        cdef ArrowSchema* child = children[i]
        return <uintptr_t>child

cdef class CArrayChildren:
    cdef CArray _parent
    cdef int64_t _length

    def __init__(self, CArray parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")

        return CArray(self._parent, self._child_addr(k))

    cdef _child_addr(self, int64_t i):
        cdef ArrowArray** children = self._parent._ptr.children
        cdef ArrowArray* child = children[i]
        return <uintptr_t>child

cdef class CArrayViewChildren:
    cdef CArrayView _parent
    cdef int64_t _length

    def __init__(self, CArrayView parent):
        self._parent = parent
        self._length = parent._ptr.n_children

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        k = int(k)
        if k < 0 or k >= self._length:
            raise IndexError(f"{k} out of range [0, {self._length})")

        return CArrayView(self._parent, self._child_addr(k), self._parent._array)

    cdef _child_addr(self, int64_t i):
        cdef ArrowArrayView** children = self._parent._ptr.children
        cdef ArrowArrayView* child = children[i]
        return <uintptr_t>child
