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

from typing import Any, Callable, List, Sequence, Tuple, Union

from nanoarrow._lib import CArrayView, CArrowType, CBuffer, CBufferBuilder
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.c_schema import c_schema_view
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator
from nanoarrow.schema import Type


def to_pylist(obj, schema=None) -> List:
    """Convert ``obj`` to a ``list()` of Python objects

    Computes an identical value to ``list(iterator.iter_py())`` but is several
    times faster.

    Paramters
    ---------
    obj : array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow import visitor
    >>> array = na.c_array([1, 2, 3], na.int32())
    >>> visitor.to_pylist(array)
    [1, 2, 3]
    """
    return ListBuilder.visit(obj, schema)


def to_columns(obj, schema=None, handle_nulls=None) -> Tuple[List[str], List[Sequence]]:
    """Convert ``obj`` to a ``list()` of sequences

    Converts a stream of struct arrays into its column-wise representation
    such that each column is either a contiguous buffer or a ``list()``.

    Paramters
    ---------
    obj : array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow import visitor
    >>> import pyarrow as pa
    >>> array = pa.record_batch([pa.array([1, 2, 3])], names=["col1"])
    >>> names, columns = visitor.to_columns(array)
    >>> names
    ['col1']
    >>> columns
    [nanoarrow.c_lib.CBuffer(int64[24 b] 1 2 3)]
    """
    return ColumnsBuilder.visit(obj, schema, handle_nulls=handle_nulls)


def nulls_forbid() -> Callable[[CBuffer, Sequence], Sequence]:
    """Erroring null handler

    A null handler that errors when it encounters nulls.

    Examples
    --------

    >>> from nanoarrow import visitor
    >>> import numpy as np
    >>> handler = visitor.nulls_forbid()
    >>> data = np.array([1, 2, 3], np.int32)
    >>> handler(np.array([], np.bool_), data)
    array([1, 2, 3], dtype=int32)
    >>> handler(np.array([True, False, True], np.bool_), data)
    Traceback (most recent call last):
    ...
    ValueError: Null present with null_handler=nulls_forbid()
    """

    def handle(is_valid, data):
        if len(is_valid) > 0:
            raise ValueError("Null present with null_handler=nulls_forbid()")

        return data

    return handle


def nulls_debug() -> Callable[[CBuffer, Sequence], Tuple[CBuffer, Sequence]]:
    """Debugging null handler

    A null handler that returns its input.

    Examples
    --------

    >>> from nanoarrow import visitor
    >>> import numpy as np
    >>> handler = visitor.nulls_debug()
    >>> data = np.array([1, 2, 3], np.int32)
    >>> handler(np.array([], np.bool_), data)
    (array([], dtype=bool), array([1, 2, 3], dtype=int32))
    >>> handler(np.array([True, False, True], np.bool_), data)
    (array([ True, False,  True]), array([1, 2, 3], dtype=int32))
    """

    def handle(is_valid, data):
        return is_valid, data

    return handle


def nulls_as_sentinel(sentinel=None):
    """Sentinel null handler

    A null handler that assigns a sentinel to null values. This is
    done using numpy using the expression ``data[~is_valid] = sentinel``.
    The default sentinel value will result in ``nan`` assigned to null
    values in numeric and boolean outputs.

    Parameters
    ----------
    sentinel : scalar, optional
        The value with which nulls should be replaced.

    Examples
    --------

    >>> from nanoarrow import visitor
    >>> import numpy as np
    >>> handler = visitor.nulls_as_sentinel()
    >>> data = np.array([1, 2, 3], np.int32)
    >>> handler(np.array([], np.bool_), data)
    array([1, 2, 3], dtype=int32)
    >>> handler(np.array([True, False, True], np.bool_), data)
    array([ 1., nan,  3.])
    >>> handler = visitor.nulls_as_sentinel(-999)
    >>> handler(np.array([True, False, True], np.bool_), data)
    array([   1, -999,    3], dtype=int32)
    """
    import numpy as np

    def handle(is_valid, data):
        is_valid = np.array(is_valid, copy=False)
        data = np.array(data, copy=False)

        if len(is_valid) > 0:
            out_type = np.result_type(data, sentinel)
            data = np.array(data, dtype=out_type, copy=True)
            data[~is_valid] = sentinel
            return data
        else:
            return data

    return handle


class ArrayStreamVisitor(ArrayViewBaseIterator):
    """Compute a value from one or more arrays in an ArrowArrayStream

    This class supports a (currently internal) pattern for building
    output from a zero or more arrays in a stream.

    """

    @classmethod
    def visit(cls, obj, schema=None, total_elements=None, **kwargs):
        """Visit all chunks in ``obj`` as a :func:`c_array_stream`."""

        if total_elements is None and hasattr(obj, "__len__"):
            total_elements = len(obj)

        with c_array_stream(obj, schema=schema) as stream:
            visitor = cls(stream._get_cached_schema(), **kwargs)
            visitor.begin(total_elements)

            visitor_set_array = visitor._set_array
            visit_chunk_view = visitor.visit_chunk_view
            array_view = visitor._array_view

            for array in stream:
                visitor_set_array(array)
                visit_chunk_view(array_view)

        return visitor.finish()

    def begin(self, total_elements: Union[int, None] = None):
        """Called after the schema has been resolved but before any
        chunks have been visited. If the total number of elements
        (i.e., the sum of all chunk lengths) is known, it is provided here.
        """
        pass

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        """Called exactly one for each chunk seen."""
        pass

    def finish(self) -> Any:
        """Called exactly once after all chunks have been visited."""
        return None


class ListBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, iterator_cls=PyIterator, array_view=None):
        super().__init__(schema, array_view=array_view)

        # Ensure that self._iterator._array_view is self._array_view
        self._iterator = iterator_cls(schema, array_view=self._array_view)

    def begin(self, total_elements: Union[int, None] = None):
        self._lst = []

    def visit_chunk_view(self, array_view: CArrayView):
        # The constructor here ensured that self._iterator._array_view
        # is populated when self._set_array() is called.
        self._lst.extend(self._iterator)

    def finish(self) -> List:
        return self._lst


class ColumnsBuilder(ArrayStreamVisitor):
    def __init__(self, schema, handle_nulls=None, *, array_view=None):
        super().__init__(schema, array_view=array_view)

        if self.schema.type != Type.STRUCT:
            raise ValueError("ColumnsBuilder can only be used on a struct array")

        # Resolve the appropriate visitor for each column
        self._child_visitors = []
        for child_schema, child_array_view in zip(
            self._schema.children, self._array_view.children
        ):
            self._child_visitors.append(
                self._resolve_child_visitor(
                    child_schema, child_array_view, handle_nulls
                )
            )

    def _resolve_child_visitor(self, child_schema, child_array_view, handle_nulls):
        cls, kwargs = _resolve_column_builder_cls(child_schema, handle_nulls)
        return cls(child_schema, **kwargs, array_view=child_array_view)

    def begin(self, total_elements: Union[int, None] = None) -> None:
        for child_visitor in self._child_visitors:
            child_visitor.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> Any:
        if array_view.null_count > 0:
            raise ValueError("null_count > 0 encountered in ColumnsBuilder")

        for child_visitor, child_array_view in zip(
            self._child_visitors, array_view.children
        ):
            child_visitor.visit_chunk_view(child_array_view)

    def finish(self) -> Tuple[List[str], List[Sequence]]:
        return [v.schema.name for v in self._child_visitors], [
            v.finish() for v in self._child_visitors
        ]


class BufferColumnBuilder(ArrayStreamVisitor):
    def begin(self, total_elements: Union[int, None]):
        self._builder = CBufferBuilder()
        self._builder.set_format(self._schema_view.buffer_format)

        if total_elements is not None:
            element_size_bits = self._schema_view.layout.element_size_bits[1]
            element_size_bytes = element_size_bits // 8
            self._builder.reserve_bytes(total_elements * element_size_bytes)

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        builder = self._builder
        offset, length = array_view.offset, array_view.length
        dst_bytes = length * builder.itemsize

        builder.reserve_bytes(dst_bytes)
        array_view.buffer(1).copy_into(builder, offset, length, len(builder))
        builder.advance(dst_bytes)

    def finish(self) -> Any:
        return self._builder.finish()


class BooleanColumnBuilder(ArrayStreamVisitor):
    def begin(self, total_elements: Union[int, None]):
        self._builder = CBufferBuilder()
        self._builder.set_format("?")

        if total_elements is not None:
            self._builder.reserve_bytes(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        builder = self._builder
        offset, length = array_view.offset, array_view.length
        builder.reserve_bytes(length)
        array_view.buffer(1).unpack_bits_into(builder, offset, length, len(builder))
        builder.advance(length)

    def finish(self) -> Any:
        return self._builder.finish()


class NullableColumnBuilder(ArrayStreamVisitor):
    def __init__(
        self,
        schema,
        column_builder_cls=BufferColumnBuilder,
        handle_nulls: Union[Callable[[CBuffer, Sequence], Any], None] = None,
        *,
        array_view=None
    ):
        super().__init__(schema, array_view=array_view)
        self._column_builder = column_builder_cls(schema, array_view=self._array_view)

        if handle_nulls is None:
            self._handle_nulls = nulls_forbid()
        else:
            self._handle_nulls = handle_nulls

    def begin(self, total_elements: Union[int, None]):
        self._builder = CBufferBuilder()
        self._builder.set_format("?")
        self._length = 0

        self._column_builder.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        offset, length = array_view.offset, array_view.length

        builder = self._builder
        chunk_contains_nulls = array_view.null_count != 0
        bitmap_allocated = len(builder) > 0

        if chunk_contains_nulls:
            current_length = self._length
            if not bitmap_allocated:
                self._fill_valid(current_length)

            builder.reserve_bytes(length)
            array_view.buffer(0).unpack_bits_into(
                builder, offset, length, current_length
            )
            builder.advance(length)

        elif bitmap_allocated:
            self._fill_valid(length)

        self._length += length
        self._column_builder.visit_chunk_view(array_view)

    def finish(self) -> Any:
        is_valid = self._builder.finish()
        column = self._column_builder.finish()
        return self._handle_nulls(is_valid, column)

    def _fill_valid(self, length):
        builder = self._builder
        builder.reserve_bytes(length)
        out_start = len(builder)
        memoryview(builder)[out_start : out_start + length] = b"\x01" * length
        builder.advance(length)


def _resolve_column_builder_cls(schema, handle_nulls=None):
    schema_view = c_schema_view(schema)

    if schema_view.nullable:
        if schema_view.type_id == CArrowType.BOOL:
            return NullableColumnBuilder, {
                "column_builder_cls": BooleanColumnBuilder,
                "handle_nulls": handle_nulls,
            }
        elif schema_view.buffer_format is not None:
            return NullableColumnBuilder, {
                "column_builder_cls": BufferColumnBuilder,
                "handle_nulls": handle_nulls,
            }
        else:
            return ListBuilder, {}
    else:

        if schema_view.type_id == CArrowType.BOOL:
            return BooleanColumnBuilder, {}
        elif schema_view.buffer_format is not None:
            return BufferColumnBuilder, {}
        else:
            return ListBuilder, {}
