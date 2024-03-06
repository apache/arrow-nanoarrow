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

from functools import cached_property
from typing import Iterable

from nanoarrow._lib import CArray, CMaterializedArrayStream, CScalar
from nanoarrow.c_lib import c_array, c_array_stream
from nanoarrow.iterator import iterator
from nanoarrow.schema import Schema


class Scalar:
    """Generic wrapper around an :class:`Array` element

    This class exists to provide a generic implementation of
    array-like indexing for the :class:`Array`. These objects
    can currently only be created by extracting an element from
    an :class:`Array`.

    Note that it is rarely efficient to iterate over Scalar objects:
    use the iterators in :mod:`nanoarrow.iterator` to more effectively
    iterate over an :class:`Array`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> array = na.array([1, 2, 3], na.int32())
    >>> array[0]
    Scalar<INT32> 1
    >>> array[0].as_py()
    1
    >>> array[0].schema
    Schema(INT32)
    """

    def __init__(self, obj):
        if not isinstance(obj, CScalar):
            raise TypeError(
                f"Can't create Scalar from object of class {type(obj).__name__}"
            )
        self._c_scalar = obj
        self._schema = None

    @property
    def schema(self) -> Schema:
        """Get the schema (data type) of this scalar"""
        if self._schema is None:
            self._schema = Schema(self._c_scalar.schema)
        return self._schema

    def as_py(self):
        """Get the Python object representation of this scalar"""
        return next(iterator(self._c_scalar))

    def __repr__(self) -> str:
        width_hint = 80
        prefix = f"Scalar<{self.schema.type.name}> "
        width_hint -= len(prefix)

        py_repr = repr(self.as_py())
        if len(py_repr) > width_hint:
            py_repr = py_repr[: (width_hint - 3)] + "..."
        return f"{prefix}{py_repr}"


class Array:
    def __init__(self, obj, schema=None) -> None:
        if isinstance(obj, Array) and schema is None:
            self._data = obj._data
            return

        if isinstance(obj, CArray) and schema is None:
            self._data = CMaterializedArrayStream.from_c_array(obj)
            return

        with c_array_stream(obj, schema=schema) as stream:
            self._data = CMaterializedArrayStream.from_c_array_stream(stream)

    def __arrow_c_stream__(self, requested_schema=None):
        return self._data.__arrow_c_stream__(requested_schema=requested_schema)

    def __arrow_c_array__(self, requested_schema=None):
        if self._data.n_arrays == 0:
            return c_array([], schema=self._data.schema).__arrow_c_array__(
                requested_schema=requested_schema
            )
        elif self._data.n_arrays == 1:
            return self._data.array(0).__arrow_c_array__(
                requested_schema=requested_schema
            )

        raise ValueError(
            f"Can't export Array with {self._data.n_arrays} chunks to ArrowArray"
        )

    @cached_property
    def schema(self) -> Schema:
        return Schema(self._data.schema)

    @property
    def n_chunks(self) -> int:
        return self._data.n_arrays

    @property
    def chunks(self) -> Iterable:
        for array in self._data.arrays:
            yield Array(array)

    def chunk(self, i):
        return Array(self._data.array(i))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, k) -> Scalar:
        scalar = Scalar(self._data[k])
        scalar._schema = self.schema
        return scalar

    def __iter__(self) -> Iterable[Scalar]:
        for c_scalar in self._data:
            scalar = Scalar(c_scalar)
            scalar._schema = self.schema
            yield scalar

    def __repr__(self) -> str:
        width_hint = 80
        n_items = 10
        lines = [f"Array<{self.schema.type.name}>[{len(self)}]"]

        for i, item in enumerate(self):
            if i >= n_items:
                break
            py_repr = repr(item.as_py())
            if len(py_repr) > width_hint:
                py_repr = py_repr[: (width_hint - 3)] + "..."
            lines.append(py_repr)

        n_more_items = len(self) - n_items
        if n_more_items > 1:
            lines.append(f"...and {n_more_items} more items")
        elif n_more_items > 0:
            lines.append(f"...and {n_more_items} more item")

        return "\n".join(lines)


def array(obj, schema=None) -> Array:
    """Create a nanoarrow Array

    The :class:`Array` class is nanoarrow's high-level in-memory array
    representation, encompasing the role of PyArrow's ``Array``,
    ``ChunkedArray``, ``RecordBatch``, and ``Table``. This scope maps
    to that of a fully-consumed ``ArrowArrayStream`` as represented by
    the Arrow C Stream interface.

    Note that an :class:`Array` is not necessarily contiguous in memory (i.e.,
    it may consist of zero or more ``ArrowArray``s).

    Parameters
    ----------
    obj : array or array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.array([1, 2, 3], na.int32())
    Array<INT32>[3]
    1
    2
    3
    """
    return Array(obj, schema=schema)
