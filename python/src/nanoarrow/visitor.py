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

from typing import Any, List, Sequence, Tuple, Union

from nanoarrow._lib import CArrayView
from nanoarrow.c_array_stream import c_array_stream
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


def to_columns(obj, schema=None) -> Tuple[List[str], List[Sequence]]:
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
    ["col1"]
    >>> columns
    [[1, 2, 3]]
    """
    return ColumnsBuilder.visit(obj, schema)


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
    def __init__(self, schema, *, iterator_cls=PyIterator, _array_view=None):
        super().__init__(schema, _array_view=_array_view)

        # Ensure that self._iterator._array_view is self._array_view
        self._iterator = iterator_cls(schema, _array_view=self._array_view)

    def begin(self, total_elements: Union[int, None] = None):
        self._lst = []

    def visit_chunk_view(self, array_view: CArrayView):
        # The constructor here ensured that self._iterator._array_view
        # is populated when self._set_array() is called.
        self._lst.extend(self._iterator)

    def finish(self) -> List:
        return self._lst


class ColumnsBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)

        if self.schema.type != Type.STRUCT:
            raise ValueError("ColumnsBuilder can only be used on a struct array")

        # Resolve the appropriate visitor for each column
        self._child_visitors = []
        for child_schema, child_array_view in zip(
            self._schema.children, self._array_view.children
        ):
            self._child_visitors.append(
                self._resolve_child_visitor(child_schema, child_array_view)
            )

    def _resolve_child_visitor(self, child_schema, child_array_view):
        # TODO: Resolve more efficient column builders for single-buffer types
        return ListBuilder(child_schema, _array_view=child_array_view)

    def begin(self, total_elements: Union[int, None] = None) -> None:
        for child_visitor in self._child_visitors:
            child_visitor.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> Any:
        for child_visitor, child_array_view in zip(
            self._child_visitors, array_view.children
        ):
            child_visitor.visit_chunk_view(child_array_view)

    def finish(self) -> Tuple[List[str], List[Sequence]]:
        return [v.schema.name for v in self._child_visitors], [
            v.finish() for v in self._child_visitors
        ]
