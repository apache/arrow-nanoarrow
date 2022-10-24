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

from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t, uint8_t, uintptr_t
from cython.operator cimport dereference as deref
from cpython cimport PyObject, Py_INCREF

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


cdef class Array:

    cdef:
        ArrowArray* array_ptr
        ArrowSchema* schema_ptr
        bint own_data
        bint own_ptrs
        ArrowArrayView array_view
        object parent

    def __init__(self):
        raise TypeError("Do not call constructor directly")

    cdef void init(self, ArrowArray* array_ptr, ArrowSchema* schema_ptr) except *:
        self.array_ptr = array_ptr
        self.schema_ptr = schema_ptr
        self.own_data = True

    def __dealloc__(self):
        if self.own_data:
            self.array_ptr.release(self.array_ptr)
            self.schema_ptr.release(self.schema_ptr)
        if self.own_ptrs:
            if self.array_ptr is not NULL:
                free(self.array_ptr)
                self.array_ptr = NULL
            if self.schema_ptr is not NULL:
                free(self.schema_ptr)
                self.schema_ptr = NULL
        self.parent = None

    @property
    def format(self):
        cdef const char* format_string = deref(self.schema_ptr).format
        if format_string == NULL:
            return None
        else:
            return format_string.decode('utf8')

    @classmethod
    def from_pyarrow(cls, arr):
        cdef ArrowSchema *schema = <ArrowSchema *>malloc(sizeof(ArrowSchema))
        cdef ArrowArray *array = <ArrowArray *>malloc(sizeof(ArrowArray))

        arr._export_to_c(<uintptr_t> array, <uintptr_t> schema)
        cdef Array self = Array.__new__(Array)
        self.init(array, schema)
        self.own_ptrs = True
        self.parent = arr
        return self

    def to_numpy(self):
        return _as_numpy_array(self)


def _as_numpy_array(Array arr):
    cdef ArrowArray* array = arr.array_ptr
    cdef ArrowArrayView array_view
    cdef ArrowError error
    cdef ArrowErrorCode ret_code

    ret_code = ArrowArrayViewInitFromSchema(&array_view, arr.schema_ptr, &error)
    if ret_code != 0:
        msg = ArrowErrorMessage(&error).decode('utf8')
        raise Exception("Could not create view: {} (error code {})".format(msg, int(ret_code)))

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
    result.base = <PyObject*> arr
    Py_INCREF(arr)

    return result
