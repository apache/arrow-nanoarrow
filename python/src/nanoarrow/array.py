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
from typing import Iterable, Tuple

from nanoarrow._lib import DEVICE_CPU, CArray, CBuffer, CMaterializedArrayStream, Device
from nanoarrow.c_lib import c_array, c_array_stream, c_array_view
from nanoarrow.iterator import iter_py, iter_tuples
from nanoarrow.schema import Schema

from nanoarrow import _repr_utils


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
    >>> array = na.Array([1, 2, 3], na.int32())
    >>> array[0]
    Scalar<int32> 1
    >>> array[0].as_py()
    1
    >>> array[0].schema
    Schema(INT32)
    """

    def __init__(self):
        # Private constructor
        self._c_array = None
        self._offset = None
        self._schema = None
        self._device = None

    @property
    def device(self) -> Device:
        return self._device

    @property
    def schema(self) -> Schema:
        """Get the schema (data type) of this scalar"""
        return self._schema

    def as_py(self):
        """Get the Python object representation of this scalar"""
        return next(iter_py(self))

    def to_string(self, width_hint=80) -> str:
        c_schema_string = _repr_utils.c_schema_to_string(
            self._c_array.schema, width_hint // 4
        )

        prefix = f"Scalar<{c_schema_string}> "
        width_hint -= len(prefix)

        py_repr = repr(self.as_py())
        if len(py_repr) > width_hint:
            py_repr = py_repr[: (width_hint - 3)] + "..."
        return f"{prefix}{py_repr}"

    def __repr__(self) -> str:
        return self.to_string()

    def __arrow_c_array__(self, requested_schema=None):
        array = self._c_array[self._offset : (self._offset + 1)]
        return array.__arrow_c_array__(requested_schema=requested_schema)


class Array:
    """High-level in-memory Array representation

    The Array is nanoarrow's high-level in-memory array representation whose
    scope maps to that of a fully-consumed ArrowArrayStream in the Arrow C Data
    interface. See :func:`array` for class details.

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
    device : Device, optional
        The device associated with the buffers held by this Array.
        Defaults to the CPU device.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.Array([1, 2, 3], na.int32())
    nanoarrow.Array<int32>[3]
    1
    2
    3
    """

    def __init__(self, obj, schema=None, device=None) -> None:
        if device is None:
            self._device = DEVICE_CPU
        elif isinstance(device, Device):
            self._device = device
        else:
            raise TypeError("device must be Device")

        if isinstance(obj, CMaterializedArrayStream) and schema is None:
            self._data = obj
            return

        if isinstance(obj, Array) and schema is None:
            self._data = obj._data
            return

        if isinstance(obj, CArray) and schema is None:
            self._data = CMaterializedArrayStream.from_c_array(obj)
            return

        with c_array_stream(obj, schema=schema) as stream:
            self._data = CMaterializedArrayStream.from_c_array_stream(stream)

    def _assert_one_chunk(self, op):
        if self._data.n_arrays != 1:
            raise ValueError(f"Can't {op} with non-contiguous Array")

    def _assert_cpu(self, op):
        if self._device != DEVICE_CPU:
            raise ValueError(f"Can't {op} with Array on non-CPU device")

    def __arrow_c_stream__(self, requested_schema=None):
        self._assert_cpu("export ArrowArrayStream")
        return self._data.__arrow_c_stream__(requested_schema=requested_schema)

    def __arrow_c_array__(self, requested_schema=None):
        self._assert_cpu("export ArrowArray")

        if self._data.n_arrays == 0:
            return c_array([], schema=self._data.schema).__arrow_c_array__(
                requested_schema=requested_schema
            )
        elif self._data.n_arrays == 1:
            return self._data.array(0).__arrow_c_array__(
                requested_schema=requested_schema
            )

        self._assert_one_chunk("export ArrowArray")

    @property
    def device(self) -> Device:
        """Get the device on which the buffers for this array are allocated.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.device
        <nanoarrow.device.Device>
        - device_type: CPU <1>
        - device_id: -1
        """
        return self._device

    @cached_property
    def schema(self) -> Schema:
        """Get the schema (data type) of this Array"""
        return Schema(self._data.schema)

    @property
    def n_buffers(self) -> int:
        """Get the number of buffers in each chunk of this Array.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.n_buffers
        2
        """
        return self.schema._c_schema_view.layout.n_buffers

    def buffer(self, i: int) -> CBuffer:
        """Access a single buffer of a contiguous array.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.buffer(1)
        nanoarrow.c_lib.CBufferView(int32[12 b] 1 2 3)
        """
        return self.buffers[i]

    @cached_property
    def buffers(self) -> Tuple[CBuffer]:
        """Access buffers of a contiguous array.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> for buffer in array.buffers:
        ...     print(buffer)
        nanoarrow.c_lib.CBufferView(bool[0 b] )
        nanoarrow.c_lib.CBufferView(int32[12 b] 1 2 3)
        """
        view = c_array_view(self)
        return tuple(view.buffers)

    def iter_buffers(self) -> Iterable[Tuple[CBuffer]]:
        """Iterate over buffers of each chunk in this Array.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> for data, validity in array.iter_buffers():
        ...     print(data)
        ...     print(validity)
        nanoarrow.c_lib.CBufferView(bool[0 b] )
        nanoarrow.c_lib.CBufferView(int32[12 b] 1 2 3)
        """
        # Could be more efficient using the iterator.ArrayViewIterator
        for chunk in self.iter_chunks():
            yield chunk.buffers

    @property
    def n_children(self) -> int:
        """Get the number of children for an Array of this type.

        Examples
        --------

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])],
        ...     names=["col1", "col2"]
        ... )
        >>> array = na.Array(batch)
        >>> array.n_children
        2
        """
        return self._data.schema.n_children

    def child(self, i: int):
        """Borrow a child Array from its parent.

        Parameters
        ----------
        i : int
            The index of the child to return.

        Examples
        --------

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])],
        ...     names=["col1", "col2"]
        ... )
        >>> array = na.Array(batch)
        >>> array.child(1)
        nanoarrow.Array<string>[3]
        'a'
        'b'
        'c'
        """
        return Array(self._data.child(i), device=self._device)

    def iter_children(self) -> Iterable:
        """Iterate over children of this Array

        Examples
        --------

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])],
        ...     names=["col1", "col2"]
        ... )
        >>> array = na.Array(batch)
        >>> for child in array.iter_children():
        ...     print(child)
        nanoarrow.Array<int64>[3]
        1
        2
        3
        nanoarrow.Array<string>[3]
        'a'
        'b'
        'c'
        """
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def n_chunks(self) -> int:
        """Get the number of chunks in the underlying representation of this Array.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.n_chunks
        1
        """
        return self._data.n_arrays

    def chunk(self, i: int):
        """Extract a single contiguous Array from the underlying representation.

        Parameters
        ----------
        i : int
            The index of the chunk to extract.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.chunk(0)
        nanoarrow.Array<int32>[3]
        1
        2
        3
        """
        return Array(self._data.array(i), device=self._device)

    def iter_chunks(self) -> Iterable:
        """Iterate over Arrays in the underlying representation whose buffers are
        contiguous in memory.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> for chunk in array.iter_chunks():
        ...     print(chunk)
        nanoarrow.Array<int32>[3]
        1
        2
        3
        """
        for array in self._data.arrays:
            yield Array(array, device=self._device)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, k) -> Scalar:
        scalar = Scalar()
        scalar._c_array, scalar._offset = self._data[k]
        scalar._schema = self.schema
        scalar._device = self._device
        return scalar

    def iter_scalar(self) -> Iterable[Scalar]:
        """Iterate over items as Scalars

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> for item in array.iter_scalar():
        ...     print(item)
        Scalar<int32> 1
        Scalar<int32> 2
        Scalar<int32> 3
        """
        for carray, offset in self._data:
            scalar = Scalar()
            scalar._c_array = carray
            scalar._offset = offset
            scalar._schema = self.schema
            scalar._device = self._device
            yield scalar

    def iter_py(self) -> Iterable:
        """Iterate over the default Python representation of each element.

        Examples
        --------

        >>> import nanoarrow as na
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> for item in array.iter_py():
        ...     print(item)
        1
        2
        3
        """
        return iter_py(self)

    def iter_tuples(self) -> Iterable[Tuple]:
        """Iterate over rows of a struct array as tuples.

        Examples
        --------

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])],
        ...     names=["col1", "col2"]
        ... )
        >>> array = na.Array(batch)
        >>> for item in array.iter_tuples():
        ...     print(item)
        (1, 'a')
        (2, 'b')
        (3, 'c')
        """
        return iter_tuples(self)

    def __iter__(self):
        raise NotImplementedError(
            "Use iter_scalar(), iter_py(), or iter_tuples() "
            "to iterate over elements of this Array"
        )

    def to_string(self, width_hint=80, items_hint=10) -> str:
        cls_name = _repr_utils.make_class_label(self, module="nanoarrow")
        len_text = f"[{len(self)}]"
        c_schema_string = _repr_utils.c_schema_to_string(
            self._data.schema, width_hint - len(cls_name) - len(len_text) - 2
        )

        lines = [f"{cls_name}<{c_schema_string}>{len_text}"]

        for i, item in enumerate(self.iter_py()):
            if i >= items_hint:
                break
            py_repr = repr(item)
            if len(py_repr) > width_hint:
                py_repr = py_repr[: (width_hint - 3)] + "..."
            lines.append(py_repr)

        n_more_items = len(self) - items_hint
        if n_more_items > 1:
            lines.append(f"...and {n_more_items} more items")
        elif n_more_items > 0:
            lines.append(f"...and {n_more_items} more item")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_string()
