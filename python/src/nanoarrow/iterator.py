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

from nanoarrow.c_lib import (
    CArrayView,
    CArrowType,
    c_array,
    c_array_stream,
    c_schema,
    c_schema_view,
)


def iteritems(obj):
    if hasattr(obj, "__arrow_c_stream__"):
        return _iteritems_stream(obj)

    obj = c_array(obj)
    iterator = ItemsIterator(obj.schema)
    iterator._set_array(obj)
    return iterator._iter1(0, obj.length)


def _iteritems_stream(obj):
    with c_array_stream(obj) as stream:
        iterator = ItemsIterator(stream._get_cached_schema())
        for array in stream:
            iterator._set_array(array)
            yield from iterator._iter1(0, array.length)


def itertuples(obj):
    if hasattr(obj, "__arrow_c_stream__"):
        return _itertuples_stream(obj)

    obj = c_array(obj)
    iterator = RowTupleIterator(obj.schema)
    iterator._set_array(obj)
    return iterator._iter1(0, obj.length)


def _itertuples_stream(obj):
    with c_array_stream(obj) as stream:
        iterator = RowTupleIterator(stream._get_cached_schema())
        for array in stream:
            iterator._set_array(array)
            yield from iterator._iter1(0, array.length)


class ArrayViewIterator:
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


class ItemsIterator(ArrayViewIterator):
    def _iter1(self, offset, length):
        schema_view = self._schema_view

        nullable = self._contains_nulls()
        type_id = schema_view.type_id
        key = nullable, type_id
        if key not in _ITEMS_ITER_LOOKUP:
            raise KeyError(f"Can't resolve iterator for type '{schema_view.type}'")

        factory = getattr(self, _ITEMS_ITER_LOOKUP[key])
        return factory(offset, length)

    def _struct_tuple_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        return zip(*(child._iter1(offset, length) for child in self._children))

    def _nullable_struct_tuple_iter(self, offset, length):
        view = self._array_view
        for is_valid, item in zip(
            view.buffer(0).elements(view.offset + offset, length),
            self._struct_tuple_iter(offset, length),
        ):
            yield item if is_valid else None

    def _struct_iter(self, offset, length):
        names = self._child_names
        tuples = self._struct_tuple_iter(offset, length)
        for item in tuples:
            yield {key: val for key, val in zip(names, item)}

    def _nullable_struct_iter(self, offset, length):
        view = self._array_view
        for is_valid, item in zip(
            view.buffer(0).elements(view.offset + offset, length),
            self._struct_iter(offset, length),
        ):
            yield item if is_valid else None

    def _list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        child = self._children[0]
        for start, end in zip(offsets[:-1], offsets[1:]):
            yield list(child._iter1(start, end - start))

    def _nullable_list_iter(self, offset, length):
        view = self._array_view
        for is_valid, item in zip(
            view.buffer(0).elements(view.offset + offset, length),
            self._list_iter(offset, length),
        ):
            yield item if is_valid else None

    def _fixed_size_list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        child = self._children[0]
        fixed_size = view.layout.child_size_elements

        for start in range(offset, offset + (fixed_size * length), fixed_size):
            yield list(child._iter1(start, fixed_size))

    def _nullable_fixed_size_list_iter(self, offset, length):
        view = self._array_view
        for is_valid, item in zip(
            view.buffer(0).elements(view.offset + offset, length),
            self._fixed_size_list_iter(offset, length),
        ):
            yield item if is_valid else None

    def _string_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        data = memoryview(view.buffer(2))
        for start, end in zip(offsets[:-1], offsets[1:]):
            yield str(data[start:end], "UTF-8")

    def _nullable_string_iter(self, offset, length):
        view = self._array_view
        validity, offsets, data = view.buffers
        offset += view.offset
        offsets = memoryview(offsets)[offset : (offset + length + 1)]
        data = memoryview(data)
        for is_valid, start, end in zip(
            validity.elements(offset, length), offsets[:-1], offsets[1:]
        ):
            if is_valid:
                yield str(data[start:end], "UTF-8")
            else:
                yield None

    def _binary_iter(self, offset, length):
        view = self._array_view
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        data = memoryview(view.buffer(2))
        for start, end in zip(offsets[:-1], offsets[1:]):
            yield bytes(data[start:end])

    def _nullable_binary_iter(self, offset, length):
        view = self._array_view
        validity, offsets, data = view.buffers
        offset += view.offset
        offsets = memoryview(offsets)[offset : (offset + length + 1)]
        data = memoryview(data)
        for is_valid, start, end in zip(
            validity.elements(offset, length), offsets[:-1], offsets[1:]
        ):
            if is_valid:
                yield bytes(data[start:end])
            else:
                yield None

    def _primitive_storage_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        return iter(view.buffer(1).elements(offset, length))

    def _nullable_primitive_storage_iter(self, offset, length):
        view = self._array_view
        is_valid, data = view.buffers
        offset += view.offset
        for is_valid, item in zip(
            is_valid.elements(offset, length),
            data.elements(offset, length),
        ):
            yield item if is_valid else None


class RowTupleIterator(ItemsIterator):
    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        if self._schema_view.type != "struct":
            raise TypeError(
                "RowTupleIterator can only iterate over struct arrays ",
                f"(got '{self._schema_view.type}')",
            )

    def _make_child(self, schema, array_view):
        return ItemsIterator(schema, _array_view=array_view)

    def _iter1(self, offset, length):
        if self._contains_nulls():
            return self._nullable_struct_tuple_iter(offset, length)
        else:
            return self._struct_tuple_iter(offset, length)


_ITEMS_ITER_LOOKUP = {
    (True, CArrowType.BINARY): "_nullable_binary_iter",
    (False, CArrowType.BINARY): "_binary_iter",
    (True, CArrowType.LARGE_BINARY): "_nullable_binary_iter",
    (False, CArrowType.LARGE_BINARY): "_binary_iter",
    (True, CArrowType.STRING): "_nullable_string_iter",
    (False, CArrowType.STRING): "_string_iter",
    (True, CArrowType.LARGE_STRING): "_nullable_string_iter",
    (False, CArrowType.LARGE_STRING): "_string_iter",
    (True, CArrowType.STRUCT): "_nullable_struct_iter",
    (False, CArrowType.STRUCT): "_struct_iter",
    (True, CArrowType.LIST): "_nullable_list_iter",
    (False, CArrowType.LIST): "_list_iter",
    (True, CArrowType.LARGE_LIST): "_nullable_list_iter",
    (False, CArrowType.LARGE_LIST): "_list_iter",
    (True, CArrowType.FIXED_SIZE_LIST): "_nullable_fixed_size_list_iter",
    (False, CArrowType.FIXED_SIZE_LIST): "_fixed_size_list_iter",
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
    _ITEMS_ITER_LOOKUP[False, type_id] = "_primitive_storage_iter"
    _ITEMS_ITER_LOOKUP[True, type_id] = "_nullable_primitive_storage_iter"
