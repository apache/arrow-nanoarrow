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


from nanoarrow.c_lib import CArrowType, CArrayView, c_array, c_schema, c_schema_view


def storage(array):
    array = c_array(array)
    iterator = ArrayViewIterator(array.schema)
    iterator._set_array(array)
    return iter(iterator)


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

    def __iter__(self):
        return self._make_iter(self)

    def _make_child(self, schema, array_view):
        return ArrayViewIterator(schema, _array_view=array_view)

    def _set_array(self, array):
        self._array_view._set_array(array)

    def _make_iter(self, instance, offset=0, length=None):
        view = instance._array_view
        if length is None:
            length = view.length

        nullable = instance._contains_nulls()
        type_id = view.storage_type_id
        key = nullable, type_id
        if key not in _LOOKUP:
            raise KeyError(
                f"Can't resolve iterator factory for type '{view.storage_type}'"
            )

        factory = _LOOKUP[key]
        return factory(instance, offset, length)

    @property
    def children(self):
        return self._children

    def _contains_nulls(self):
        return (
            self._schema_view.nullable
            and len(self._array_view.buffer(0))
            and self._array_view.null_count != 0
        )


def _struct_iter(iterator, offset, length):
    view = iterator._array_view
    offset += view.offset
    return zip(
        *(iterator._make_iter(child, offset, length) for child in iterator.children)
    )


def _nullable_struct_iter(iterator, offset, length):
    view = iterator._array_view
    for is_valid, item in zip(
        view.buffer(0).elements(view.offset + offset, length),
        _struct_iter(iterator, offset, length),
    ):
        yield item if is_valid else None


def _list_iter(iterator, offset, length):
    view = iterator._array_view
    offset += view.offset
    offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
    child = iterator.children[0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        yield list(iterator._make_iter(child, start, end - start))


def _nullable_list_iter(iterator, offset, length):
    view = iterator._array_view
    for is_valid, item in zip(
        view.buffer(0).elements(view.offset + offset, length),
        _list_iter(iterator, offset, length),
    ):
        yield item if is_valid else None


def _fixed_size_list_iter(iterator, offset, length):
    view = iterator._array_view
    offset += view.offset
    child = iterator.children[0]
    fixed_size = view.layout.child_size_elements

    for start in range(offset, offset + (fixed_size * length), fixed_size):
        yield list(iterator._make_iter(child, start, fixed_size))


def _nullable_fixed_size_list_iter(iterator, offset, length):
    view = iterator._array_view
    for is_valid, item in zip(
        view.buffer(0).elements(view.offset + offset, length),
        _fixed_size_list_iter(iterator, offset, length),
    ):
        yield item if is_valid else None


def _string_iter(iterator, offset, length):
    view = iterator._array_view
    offset += view.offset
    offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
    data = memoryview(view.buffer(2))
    for start, end in zip(offsets[:-1], offsets[1:]):
        yield str(data[start:end], "UTF-8")


def _nullable_string_iter(iterator, offset, length):
    view = iterator._array_view
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


def _binary_iter(iterator, offset, length):
    view = iterator._array_view
    offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
    data = memoryview(view.buffer(2))
    for start, end in zip(offsets[:-1], offsets[1:]):
        yield bytes(data[start:end])


def _nullable_binary_iter(iterator, offset, length):
    view = iterator._array_view
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


def _primitive_storage_iter(iterator, offset, length):
    view = iterator._array_view
    offset += view.offset
    return iter(view.buffer(1).elements(offset, length))


def _nullable_primitive_storage_iter(iterator, offset, length):
    view = iterator._array_view
    is_valid, data = view.buffers
    offset += view.offset
    for is_valid, item in zip(
        is_valid.elements(offset, length),
        data.elements(offset, length),
    ):
        yield item if is_valid else None


_LOOKUP = {
    (True, CArrowType.BINARY): _nullable_binary_iter,
    (False, CArrowType.BINARY): _binary_iter,
    (True, CArrowType.LARGE_BINARY): _nullable_binary_iter,
    (False, CArrowType.LARGE_BINARY): _binary_iter,
    (True, CArrowType.STRING): _nullable_string_iter,
    (False, CArrowType.STRING): _string_iter,
    (True, CArrowType.LARGE_STRING): _nullable_string_iter,
    (False, CArrowType.LARGE_STRING): _string_iter,
    (True, CArrowType.STRUCT): _nullable_struct_iter,
    (False, CArrowType.STRUCT): _struct_iter,
    (True, CArrowType.LIST): _nullable_list_iter,
    (False, CArrowType.LIST): _list_iter,
    (True, CArrowType.LARGE_LIST): _nullable_list_iter,
    (False, CArrowType.LARGE_LIST): _list_iter,
    (True, CArrowType.FIXED_SIZE_LIST): _nullable_fixed_size_list_iter,
    (False, CArrowType.FIXED_SIZE_LIST): _fixed_size_list_iter,
}

_PRIMITIVE = [
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

for type_name in _PRIMITIVE:
    type_id = getattr(CArrowType, type_name)
    _LOOKUP[False, type_id] = _primitive_storage_iter
    _LOOKUP[True, type_id] = _nullable_primitive_storage_iter
