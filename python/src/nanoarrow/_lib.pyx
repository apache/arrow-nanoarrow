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

        return type(self._parent)(self._parent, self._child_addr(k))

    cdef _child_addr(self, int64_t i):
        cdef ArrowSchema** children = self._parent._ptr.children
        cdef ArrowSchema* child = children[i]
        return <uintptr_t>child

cdef class CSchema:
    cdef object _base
    cdef ArrowSchema* _ptr

    def __init__(self, object base, uintptr_t addr) -> None:
        self._base = base,
        self._ptr = <ArrowSchema*>addr

    def _addr(self):
        return <uintptr_t>self._ptr

    def __repr__(self) -> str:
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
        if self._ptr.format != NULL:
            return self._ptr.format.decode("UTF-8")

    @property
    def name(self):
        if self._ptr.name != NULL:
            return self._ptr.name.decode("UTF-8")

    @property
    def flags(self):
        return self._ptr.flags

    @property
    def children(self):
        return CSchemaChildren(self)
