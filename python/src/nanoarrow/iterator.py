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
from io import StringIO
from itertools import islice
from typing import Iterable, Tuple

from nanoarrow.c_lib import (
    CArrayView,
    CArrowType,
    c_array_stream,
    c_schema,
    c_schema_view,
)


def iterator(obj, schema=None) -> Iterable:
    """Iterate over items in zero or more arrays

    Returns an iterator over an array stream where each item is a
    Python representation of the next element.

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
    >>> from nanoarrow import iterator
    >>> array = na.c_array([1, 2, 3], na.int32())
    >>> list(iterator.iterator(array))
    [1, 2, 3]
    """
    return RowIterator.get_iterator(obj, schema=schema)


def itertuples(obj, schema=None) -> Iterable[Tuple]:
    """Iterate over rows in zero or more struct arrays

    Returns an iterator over an array stream of struct arrays (i.e.,
    record batches) where each item is a tuple of the items in each
    row. This is different than :func:`iterator`, which encodes struct
    columns as dictionaries.

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
    >>> from nanoarrow import iterator
    >>> import pyarrow as pa
    >>> array = pa.record_batch([pa.array([1, 2, 3])], names=["col1"])
    >>> list(iterator.itertuples(array))
    [(1,), (2,), (3,)]
    """
    return RowTupleIterator.get_iterator(obj, schema=schema)


def iterrepr(obj, schema=None, max_width: int = 80) -> Iterable[str]:
    """Iterator of reprs with bounded size for each item

    Whereas the the RowTupleIterator and the RowIterator are optimized for
    converting the entire input to Python, this iterator is defensive
    about materializing elements as Python objects and in most cases will
    not materialize a value or child value in its entirity once ``max_width``
    characters have already been materialized.

    Paramters
    ---------
    obj : array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.
    max_width : int, optional
        The maximum number of characters to include for each item.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow import iterator
    >>> array = na.c_array([1234567890, 123456789, 1234], na.int64())
    >>> for item in iterator.iterrepr(array, max_width=9):
    ...     print(item)
    123456...
    123456789
    1234
    """
    return ReprIterator.get_iterator(obj, schema, max_width=max_width)


class ArrayViewIterator:
    """Base class for iterators that use an internal ArrowArrayView
    as the basis for conversion to Python objects. Intended for internal use.
    """

    def __init__(self, schema, *, _array_view=None):
        self._schema = c_schema(schema)
        self._schema_view = c_schema_view(schema)

        if _array_view is None:
            self._array_view = CArrayView.from_schema(self._schema)
        else:
            self._array_view = _array_view

        self._children = list(
            map(self._make_child, self._schema.children, self._array_view.children)
        )

        if schema.dictionary is None:
            self._dictionary = None
        else:
            self._dictionary = self._make_child(
                self._schema.dictionary, self._array_view.dictionary
            )

    def _make_child(self, schema, array_view):
        return type(self)(schema, _array_view=array_view)

    @cached_property
    def _child_names(self):
        return [child.name for child in self._schema.children]

    def _contains_nulls(self):
        return (
            self._schema_view.nullable
            and len(self._array_view.buffer(0))
            and self._array_view.null_count != 0
        )

    def _set_array(self, array):
        self._array_view._set_array(array)
        return self


class RowIterator(ArrayViewIterator):
    """Iterate over the Python object version of values in an ArrowArrayView.
    Intended for internal use.
    """

    @classmethod
    def get_iterator(cls, obj, schema=None, **kwargs):
        with c_array_stream(obj, schema=schema) as stream:
            iterator = cls(stream._get_cached_schema(), **kwargs)
            for array in stream:
                iterator._set_array(array)
                yield from iterator._iter1(0, array.length)

    def _iter1(self, offset, length):
        type_id = self._schema_view.type_id
        if type_id not in _ITEMS_ITER_LOOKUP:
            raise KeyError(f"Can't resolve iterator for type '{self.schema_view.type}'")

        factory = getattr(self, _ITEMS_ITER_LOOKUP[type_id])
        return factory(offset, length)

    def _dictionary_iter(self, offset, length):
        dictionary = list(
            self._dictionary._iter1(0, self._dictionary._array_view.length)
        )
        for dict_index in self._primitive_iter(offset, length):
            yield None if dict_index is None else dictionary[dict_index]

    def _wrap_iter_nullable(self, validity, items):
        for is_valid, item in zip(validity, items):
            yield item if is_valid else None

    def _struct_tuple_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        items = zip(*(child._iter1(offset, length) for child in self._children))

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            return self._wrap_iter_nullable(validity, items)
        else:
            return items

    def _struct_iter(self, offset, length):
        names = self._child_names
        for item in self._struct_tuple_iter(offset, length):
            yield None if item is None else {key: val for key, val in zip(names, item)}

    def _list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset

        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]
        child = self._children[0]
        child_iter = child._iter1(starts[0], ends[-1] - starts[0])

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid, start, end in zip(validity, starts, ends):
                item = list(islice(child_iter, end - start))
                yield item if is_valid else None
        else:
            for start, end in zip(starts, ends):
                yield list(islice(child_iter, end - start))

    def _fixed_size_list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        child = self._children[0]
        fixed_size = view.layout.child_size_elements
        child_iter = child._iter1(offset * fixed_size, length * fixed_size)

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid in validity:
                item = list(islice(child_iter, fixed_size))
                yield item if is_valid else None
        else:
            for _ in range(length):
                yield list(islice(child_iter, fixed_size))

    def _string_iter(self, offset, length):
        return self._binary_view_iter(offset, length, lambda x: str(x, "UTF-8"))

    def _binary_iter(self, offset, length):
        return self._binary_view_iter(offset, length, bytes)

    def _binary_view_iter(self, offset, length, fun=lambda x: x):
        view = self._array_view
        offset += view.offset
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]
        data = memoryview(view.buffer(2))

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid, start, end in zip(validity, starts, ends):
                yield fun(data[start:end]) if is_valid else None
        else:
            for start, end in zip(starts, ends):
                yield fun(data[start:end])

    def _primitive_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        items = view.buffer(1).elements(offset, length)

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            return self._wrap_iter_nullable(validity, items)
        else:
            return iter(items)


class RowTupleIterator(RowIterator):
    """Iterate over rows of a struct array (stream) where each row is a
    tuple instead of a dictionary. This is ~3x faster and matches other
    Python concepts more closely (e.g., dbapi's cursor, pandas itertuples).
    Intended for internal use.
    """

    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        if self._schema_view.type != "struct":
            raise TypeError(
                "RowTupleIterator can only iterate over struct arrays "
                f"(got '{self._schema_view.type}')"
            )

    def _make_child(self, schema, array_view):
        return RowIterator(schema, _array_view=array_view)

    def _iter1(self, offset, length):
        return self._struct_tuple_iter(offset, length)


class ReprIterator(RowIterator):
    """Iterate over elements by materializing as little of each element
    as possible. This is not designed to be a general-purpose formatter;
    however, it is designed to allow the nanoarrow.Array class to have a
    reasonable repr() method that doesn't overflow a console or hang when
    printing arbitrary input.
    """

    def __init__(self, schema, *, _array_view=None, max_width=120, out=None):
        # Pass the same instance of out to all children. If unspecified,
        # this is the top-level iterator that will request the value from
        # out for each item.
        if out is None:
            self._out = ReprBuilder(max_width)
            self._top_level = True
        else:
            self._out = out
            self._top_level = False

        super().__init__(schema, _array_view=_array_view)

    def _make_child(self, schema, array_view):
        # Ensure children all refer to the same instance of the ReprBuilder
        return ReprIterator(schema, _array_view=array_view, out=self._out)

    def _iter1(self, offset, length):
        # The iterator resolved by the RowIterator here will yield False
        # if self._out.full(). This pattern allows arbitrary levels of
        # recursion for nested types while allowing child iterators to signal
        # to higher levels that there is no need to write any further
        # values for the top-level element.
        parent = super()._iter1(offset, length)

        # If we are at the top level, yield the (potentially abbreviated)
        # repr() string for each element. For child elements, yield
        # False if self._out.full() or True otherwise.
        if self._top_level:
            return self._repr_wrapper(parent)
        else:
            return parent

    def _repr_wrapper(self, parent):
        # A wrapper that iterates over the parent (where consuming a value of the
        # iterator writes to self._out)
        for _ in parent:
            yield self._out.finish()

    def _dictionary_iter(self, offset, length):
        # Rather than materialize the entire dictionary, materialize a single
        # element as required. This is slower for materializing many values but
        # typically this iterator will only materialize the number of rows on a
        # console at most.
        for dict_index in super()._primitive_iter(offset, length):
            if dict_index is None:
                yield self._out.write_null()
            else:
                dict_iter = self._dictionary._iter1(dict_index, 1)
                yield next(dict_iter)

    def _write_list_item(self, offset_plus_i, start, end):
        # Whereas the RowIterator resolves one iterator for child and pulls
        # end - start values from it for each element, this iterator resolves
        # length-one child iterator for each element and consumes it. This
        # pattern avoids materializing unnecessary elements.
        validity = self._array_view.buffer(0)
        item_is_valid = len(validity) == 0 or validity.element(offset_plus_i)
        if not item_is_valid:
            return self._out.write_null()

        self._out.write("[")
        for i in range(start, end):
            if i > start:
                self._out.write(", ")

            child_iter = self._children[0]._iter1(i, 1)
            if next(child_iter) is False:
                return False

        return self._out.write("]")

    def _list_iter(self, offset, length):
        offset += self._array_view.offset
        offsets = memoryview(self._array_view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]

        for i, start, end in zip(range(length), starts, ends):
            yield self._write_list_item(offset + i, start, end)

    def _fixed_size_list_iter(self, offset, length):
        offset += self._array_view.offset
        fixed_size = self._array_view.layout.child_size_elements
        for i in range(length):
            yield self._write_list_item(
                offset + i, (offset + i) * fixed_size, (offset + i + 1) * fixed_size
            )

    def _write_struct_item(self, offset_plus_i):
        # Whereas the RowIterator and RowTupleIterator materialize all child
        # iterators and always materialize a value for every child element,
        # this pattern resolves a length-one iterator for every child, consumes
        # it, and stops iterating if there are many columns. Very wide tables
        # will still incur a full loop over children at the C level (e.g., in
        # ArrowArrayViewInitFromSchema() and ArrowArrayViewSetArray()); however,
        # should not incur any Python calls in a full loop over children.
        validity = self._array_view.buffer(0)
        item_is_valid = len(validity) == 0 or validity.element(offset_plus_i)
        if not item_is_valid:
            return self._out.write_null()

        self._out.write("{")
        for i, child in enumerate(self._children):
            if i > 0:
                self._out.write(", ")

            child_name = child._schema.name
            if self._out.write(repr(child_name)) is False:
                return False

            self._out.write(": ")

            child_iter = child._iter1(offset_plus_i, 1)
            if next(child_iter) is False:
                return False

        return self._out.write("}")

    def _struct_iter(self, offset, length):
        offset += self._array_view.offset
        for i in range(length):
            yield self._write_struct_item(offset + i)

    def _string_iter(self, offset, length):
        # A variant of iterating over strings that ensures that very large
        # elements are not fully materialized as a Python object.
        memoryviews = super()._binary_view_iter(offset, length)

        # In the end-member scenario where every code point is four bytes,
        # ensure we still slice enough bytes to fill max_width. Give some
        # bytes on the end in case the last byte is the beginning of a
        # multibyte character and to ensure we never get the trailing quote
        # for an incomplete repr
        max_width_slice_bytes = (self._out._max_size + 2) * 4
        for mv in memoryviews:
            if mv is None:
                yield self._out.write_null()
            else:
                str_begin = bytes(mv[:max_width_slice_bytes]).decode()
                yield self._out.write(repr(str_begin))

    def _binary_iter(self, offset, length):
        # A variant of iterating over strings that ensures that very large
        # elements are not fully materialized as a Python object.

        memoryviews = super()._binary_view_iter(offset, length)
        # Give some extra bytes to ensure we never get a trailing ' for an
        # incomplete repr.
        max_width_slice_bytes = self._out._max_size + 4
        for mv in memoryviews:
            if mv is None:
                yield self._out.write_null()
            else:
                yield self._out.write(repr(bytes(mv[:max_width_slice_bytes])))

    def _primitive_iter(self, offset, length):
        # These types can have relatively long reprs (e.g., large int64) but they
        # are never so large that they can hang the repr (with the possible
        # exception of fixed-size binary in rare cases).
        for item in super()._primitive_iter(offset, length):
            if item is None:
                yield self._out.write_null()
            else:
                yield self._out.write(repr(item))


class ReprBuilder:
    """Stateful string builder for building repr strings

    A wrapper around io.StringIO() that keeps track of how many
    characters have been written vs some maximum size.
    """

    def __init__(self, max_size) -> None:
        self._out = StringIO()
        self._max_size = max_size
        self._size = 0

    def full(self):
        return self._size > self._max_size

    def write_null(self):
        return self.write(repr(None))

    def write(self, content):
        """Write string content. Returns True if there are not yet max_size
        characters in this items's repr and False otherwise."""
        self._out.write(content)
        self._size += len(content)
        return not self.full()

    def finish(self):
        out = self._out.getvalue()
        self._out = StringIO()
        self._size = 0

        if len(out) > self._max_size:
            return out[: (self._max_size - 3)] + "..."
        else:
            return out


_ITEMS_ITER_LOOKUP = {
    CArrowType.BINARY: "_binary_iter",
    CArrowType.LARGE_BINARY: "_binary_iter",
    CArrowType.STRING: "_string_iter",
    CArrowType.LARGE_STRING: "_string_iter",
    CArrowType.STRUCT: "_struct_iter",
    CArrowType.LIST: "_list_iter",
    CArrowType.LARGE_LIST: "_list_iter",
    CArrowType.FIXED_SIZE_LIST: "_fixed_size_list_iter",
    CArrowType.DICTIONARY: "_dictionary_iter",
}

_PRIMITIVE_TYPE_NAMES = [
    "BOOL",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "UINT32",
    "INT32",
    "UINT64",
    "INT64",
    "HALF_FLOAT",
    "FLOAT",
    "DOUBLE",
    "FIXED_SIZE_BINARY",
    "INTERVAL_MONTHS",
    "INTERVAL_DAY_TIME",
    "INTERVAL_MONTH_DAY_NANO",
    "DECIMAL128",
    "DECIMAL256",
]

for type_name in _PRIMITIVE_TYPE_NAMES:
    type_id = getattr(CArrowType, type_name)
    _ITEMS_ITER_LOOKUP[type_id] = "_primitive_iter"
