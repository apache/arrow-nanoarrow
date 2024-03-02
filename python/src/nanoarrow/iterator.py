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


def iterrepr(obj, schema=None, max_width=80):
    return ReprIterator.get_iterator(obj, schema, max_width=max_width)


class ArrayViewIterator:
    """Base class for iterators that use an internal ArrowArrayStream
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
        return self._binary_iter(offset, length, lambda x: str(x, "UTF-8"))

    def _binary_iter(self, offset, length, fun=bytes):
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


class ReprLongEnough(Exception):
    def __init__(self) -> None:
        super().__init__()


class ItemRepr:
    def __init__(self, max_size) -> None:
        self._out = StringIO()
        self._max_size = max_size
        self._size = 0

    def write(self, content):
        self._out.write(content)
        self._size += len(content)
        if self._size > self._max_size:
            raise ReprLongEnough()

    def finish(self):
        out = self._out.getvalue()
        self._out.seek(0)
        self._out.truncate()
        if len(out) > self._max_size:
            return out[: (self._max_size - 3)] + "..."
        else:
            return out


class ReprIterator(RowIterator):
    def __init__(self, schema, *, _array_view=None, max_width=80, out=None):
        super().__init__(schema, _array_view=_array_view)

        self._max_width = max_width
        if out is None:
            self._out = ItemRepr(max_width)
            self._top_level = True
        else:
            self._out = out
            self._top_level = False

    def _make_child(self, schema, array_view):
        return super()._make_child(schema, array_view)

    def _iter1(self, offset, length):
        parent = super()._iter1(offset, length)
        if self._top_level:
            return self._repr_wrapper(parent)
        else:
            return parent

    def _repr_wrapper(self, parent):
        for item in parent:
            try:
                self._out.write(repr(item))
            except ReprLongEnough:
                pass
            yield self._out.finish()

    def _dictionary_iter(self, offset, length):
        return super()._dictionary_iter(offset, length)

    def _list_iter(self, offset, length):
        return super()._list_iter(offset, length)

    def _fixed_size_list_iter(self, offset, length):
        return super()._fixed_size_list_iter(offset, length)

    def _struct_iter(self, offset, length):
        return super()._struct_iter(offset, length)

    def _string_iter(self, offset, length):
        return super()._string_iter(offset, length)

    def _binary_iter(self, offset, length, fun=bytes):
        return super()._binary_iter(offset, length, fun)

    def _primitive_iter(self, offset, length):
        return super()._primitive_iter(offset, length)


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
