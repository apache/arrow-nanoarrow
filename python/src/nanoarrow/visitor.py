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

from nanoarrow._array import CArrayView
from nanoarrow._buffer import CBuffer, CBufferBuilder
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.c_schema import c_schema_view
from nanoarrow.extension import resolve_extension
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator
from nanoarrow.schema import Type

from nanoarrow import _types


class ArrayViewVisitable:
    """Mixin class providing conversion methods based on visitors

    Can be used with classes that implement ``__arrow_c_stream__()``
    or ``__arrow_c_array__()``.
    """

    def to_pylist(self) -> List:
        """Convert to a ``list`` of Python objects

        Computes an identical value to ``list(iter_py())`` but can be much
        faster.

        Examples
        --------
        >>> import nanoarrow as na
        >>> from nanoarrow import visitor
        >>> array = na.Array([1, 2, 3], na.int32())
        >>> array.to_pylist()
        [1, 2, 3]
        """
        return ToPyListConverter.visit(self)

    def to_columns_pysequence(
        self, *, handle_nulls=None
    ) -> Tuple[List[str], List[Sequence]]:
        """Convert to a ``list`` of contiguous sequences

        Experimentally converts a stream of struct arrays into a list of contiguous
        sequences using the same logic as :meth:`to_pysequence`.

        Paramters
        ---------
        handle_nulls : callable
            A function returning a sequence based on a validity bytemap and a
            contiguous buffer of values. If the array contains no nulls, the
            validity bytemap will be ``None``. Built-in handlers include
            :func:`nulls_as_sentinel`, :func:`nulls_forbid`, and
            :func:`nulls_separate`). The default value is :func:`nulls_forbid`.

        Examples
        --------
        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        >>> names, columns = na.Array(batch).to_columns_pysequence()
        >>> names
        ['col1', 'col2']
        >>> columns
        [nanoarrow.c_buffer.CBuffer(int64[24 b] 1 2 3), ['a', 'b', 'c']]
        """
        return ToColumnsPysequenceConverter.visit(self, handle_nulls=handle_nulls)

    def to_pysequence(self, *, handle_nulls=None) -> Sequence:
        """Convert to a contiguous sequence

        Experimentally converts a stream of arrays into a columnar representation
        such that each column is either a contiguous buffer or a ``list``.
        Integer, float, and interval arrays are currently converted to their
        contiguous buffer representation; other types are returned as a list
        of Python objects. The sequences returned by :meth:`to_pysequence` are
        designed to work as input to ``pandas.Series`` and/or ``numpy.array()``.
        The default conversions are subject to change based on initial user
        feedback.

        Parameters
        ----------
        handle_nulls : callable
            A function returning a sequence based on a validity bytemap and a
            contiguous buffer of values. If the array contains no nulls, the
            validity bytemap will be ``None``. Built-in handlers include
            :func:`nulls_as_sentinel`, :func:`nulls_forbid`, and
            :func:`nulls_separate`). The default value is :func:`nulls_forbid`.

        Examples
        --------
        >>> import nanoarrow as na
        >>> na.Array([1, 2, 3], na.int32()).to_pysequence()
        nanoarrow.c_buffer.CBuffer(int32[12 b] 1 2 3)
        """
        return ToPySequenceConverter.visit(self, handle_nulls=handle_nulls)


def nulls_forbid() -> Callable[[CBuffer, Sequence], Sequence]:
    """Erroring null handler

    A null handler that errors when it encounters nulls.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.Array([1, 2, 3], na.int32()).to_pysequence(handle_nulls=na.nulls_forbid())
    nanoarrow.c_buffer.CBuffer(int32[12 b] 1 2 3)
    >>> na.Array([1, None, 3], na.int32()).to_pysequence(handle_nulls=na.nulls_forbid())
    Traceback (most recent call last):
    ...
    ValueError: Null present with null_handler=nulls_forbid()
    """

    def handle(is_valid, data):
        # the is_valid bytemap is only created if there was at least one null
        if is_valid is not None:
            raise ValueError("Null present with null_handler=nulls_forbid()")

        return data

    return handle


def nulls_as_sentinel(sentinel=None):
    """Sentinel null handler

    A null handler that assigns a sentinel to null values. This is
    done using numpy using the expression ``data[~is_valid] = sentinel``.
    The default sentinel value of ``None`` will result in float output and ``nan``
    assigned to null values for numeric and boolean inputs. This
    corresponds to numpy's handling of ``None`` in ``np.result_type()``
    and ``result[~is_valid] = None``.

    Parameters
    ----------
    sentinel : scalar, optional
        The value with which nulls should be replaced.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na_array = na.Array([1, 2, 3], na.int32())
    >>> na_array.to_pysequence(handle_nulls=na.nulls_as_sentinel())
    array([1, 2, 3], dtype=int32)
    >>> na_array = na.Array([1, None, 3], na.int32())
    >>> na_array.to_pysequence(handle_nulls=na.nulls_as_sentinel())
    array([ 1., nan,  3.])
    >>> na_array.to_pysequence(handle_nulls=na.nulls_as_sentinel(-999))
    array([   1, -999,    3], dtype=int32)
    """
    import numpy as np

    def handle(is_valid, data):
        data = np.array(data, copy=False)

        if is_valid is not None:
            is_valid = np.array(is_valid, copy=False)
            out_type = np.result_type(data, sentinel)
            data = np.array(data, dtype=out_type, copy=True)
            data[~is_valid] = sentinel
            return data
        else:
            return data

    return handle


def nulls_separate() -> Callable[[CBuffer, Sequence], Tuple[CBuffer, Sequence]]:
    """Return nulls as a tuple of is_valid, data

    A null handler that returns its components.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na_array = na.Array([1, 2, 3], na.int32())
    >>> na_array.to_pysequence(handle_nulls=na.nulls_separate())
    (None, nanoarrow.c_buffer.CBuffer(int32[12 b] 1 2 3))
    >>> na_array = na.Array([1, None, 3], na.int32())
    >>> result = na_array.to_pysequence(handle_nulls=na.nulls_separate())
    >>> result[0]
    nanoarrow.c_buffer.CBuffer(uint8[3 b] True False True)
    >>> result[1]
    nanoarrow.c_buffer.CBuffer(int32[12 b] 1 0 3)
    """

    def handle(is_valid, data):
        return is_valid, data

    return handle


class ArrayViewVisitor(ArrayViewBaseIterator):
    """Compute a value from one or more arrays as ArrowArrayViews

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


class ToPySequenceConverter(ArrayViewVisitor):
    def __init__(self, schema, handle_nulls=None, *, array_view=None):
        super().__init__(schema, array_view=array_view)
        cls, kwargs = _resolve_converter_cls(self._schema, handle_nulls=handle_nulls)
        self._visitor = cls(schema, **kwargs, array_view=self._array_view)

    def begin(self, total_elements: Union[int, None] = None):
        self._visitor.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        self._visitor.visit_chunk_view(array_view)

    def finish(self) -> Any:
        return self._visitor.finish()


class ToColumnsPysequenceConverter(ArrayViewVisitor):
    def __init__(self, schema, handle_nulls=None, *, array_view=None):
        super().__init__(schema, array_view=array_view)

        if self.schema.type != Type.STRUCT:
            raise ValueError("ToColumnListConverter can only be used on a struct array")

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
        cls, kwargs = _resolve_converter_cls(child_schema, handle_nulls)
        return cls(child_schema, **kwargs, array_view=child_array_view)

    def begin(self, total_elements: Union[int, None] = None) -> None:
        for child_visitor in self._child_visitors:
            child_visitor.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> Any:
        # This visitor does not handle nulls because it has no way to propagate these
        # into the child columns. It is designed to be used on top-level record batch
        # arrays which typically are marked as non-nullable or do not contain nulls.
        if array_view.null_count > 0:
            raise ValueError("null_count > 0 encountered in ToColumnListConverter")

        for child_visitor, child_array_view in zip(
            self._child_visitors, array_view.children
        ):
            child_visitor.visit_chunk_view(child_array_view)

    def finish(self) -> Tuple[List[str], List[Sequence]]:
        return [child.name for child in self._schema.children], [
            v.finish() for v in self._child_visitors
        ]


class ToPyListConverter(ArrayViewVisitor):
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


class ToPyBufferConverter(ArrayViewVisitor):
    def begin(self, total_elements: Union[int, None]):
        self._builder = self._make_builder()

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

    def _make_builder(self):
        return CBufferBuilder().set_format(self._schema_view.buffer_format)


class ToBooleanBufferConverter(ArrayViewVisitor):
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


class ToNullableSequenceConverter(ArrayViewVisitor):
    def __init__(
        self,
        schema,
        converter_cls=ToPyBufferConverter,
        handle_nulls: Union[Callable[[CBuffer, Sequence], Any], None] = None,
        *,
        array_view=None
    ):
        super().__init__(schema, array_view=array_view)
        self._converter = converter_cls(schema, array_view=self._array_view)

        if handle_nulls is None:
            self._handle_nulls = nulls_forbid()
        else:
            self._handle_nulls = handle_nulls

    def begin(self, total_elements: Union[int, None]):
        self._builder = CBufferBuilder()
        self._builder.set_format("?")
        self._length = 0

        self._converter.begin(total_elements)

    def visit_chunk_view(self, array_view: CArrayView) -> None:
        offset, length = array_view.offset, array_view.length

        builder = self._builder
        chunk_contains_nulls = array_view.null_count != 0
        bitmap_allocated = len(builder) > 0

        if chunk_contains_nulls:
            current_length = self._length
            if not bitmap_allocated:
                builder.write_fill(1, current_length)

            builder.reserve_bytes(length)
            array_view.buffer(0).unpack_bits_into(
                builder, offset, length, current_length
            )
            builder.advance(length)

        elif bitmap_allocated:
            builder.write_fill(1, length)

        self._length += length
        self._converter.visit_chunk_view(array_view)

    def finish(self) -> Any:
        is_valid = self._builder.finish()
        data = self._converter.finish()
        return self._handle_nulls(is_valid if len(is_valid) > 0 else None, data)


def _resolve_converter_cls(schema, handle_nulls=None):
    schema_view = c_schema_view(schema)
    ext = resolve_extension(schema_view)
    ext_converter_cls = ext.get_sequence_converter(schema) if ext else None

    if schema_view.nullable:
        if ext_converter_cls:
            return ToNullableSequenceConverter, {
                "converter_cls": ext_converter_cls,
                "handle_nulls": handle_nulls,
            }
        elif schema_view.type_id == _types.BOOL:
            return ToNullableSequenceConverter, {
                "converter_cls": ToBooleanBufferConverter,
                "handle_nulls": handle_nulls,
            }
        elif schema_view.buffer_format is not None:
            return ToNullableSequenceConverter, {
                "converter_cls": ToPyBufferConverter,
                "handle_nulls": handle_nulls,
            }
        else:
            return ToPyListConverter, {}
    else:
        if ext_converter_cls:
            return ext_converter_cls, {}
        elif schema_view.type_id == _types.BOOL:
            return ToBooleanBufferConverter, {}
        elif schema_view.buffer_format is not None:
            return ToPyBufferConverter, {}
        else:
            return ToPyListConverter, {}
