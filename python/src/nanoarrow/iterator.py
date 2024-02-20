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

from nanoarrow.c_lib import CArrowType, c_array_view


def storage(view, child_factory=None):
    if child_factory is None:
        child_factory = storage

    view = c_array_view(view)

    nullable = _array_view_nullable(view)
    type_id = view.storage_type_id
    key = nullable, type_id
    if key not in _LOOKUP:
        raise KeyError(
            f"Can't resolve iterator factory for storage type '{view.storage_type}'"
        )

    factory = _LOOKUP[key]
    return factory(view, storage)


def _struct_iter(view, child_factory):
    if view.offset != 0:
        raise NotImplementedError("Offset != 0 not implemented")

    return zip(*(child_factory(child, child_factory) for child in view.children))


def _nullable_struct_iter(view, child_factory):
    for is_valid, item in zip(
        view.buffer(0).elements(view.offset, view.length),
        _struct_iter(view, child_factory),
    ):
        yield item if is_valid else None


def _string_iter(view, child_factory):
    offsets = memoryview(view.buffer(1))
    data = memoryview(view.buffer(2))
    for start, end in zip(offsets[:-1], offsets[1:]):
        yield str(data[start:end], "UTF-8")


def _nullable_string_iter(view, child_factory):
    validity, offsets, data = view.buffers
    offsets = memoryview(offsets)[view.offset :]
    data = memoryview(data)
    for is_valid, start, end in zip(
        validity.elements(view.offset, view.length), offsets[:-1], offsets[1:]
    ):
        if is_valid:
            yield str(data[start:end], "UTF-8")
        else:
            yield None


def _binary_iter(view, child_factory):
    offsets = memoryview(view.buffer(1))[view.offset :]
    data = memoryview(view.buffer(2))
    for start, end in zip(offsets[:-1], offsets[1:]):
        yield bytes(data[start:end])


def _nullable_binary_iter(view, child_factory):
    validity, offsets, data = view.buffers
    offsets = memoryview(offsets)[view.offset :]
    data = memoryview(data)
    for is_valid, start, end in zip(validity.elements(), offsets[:-1], offsets[1:]):
        if is_valid:
            yield bytes(data[start:end])
        else:
            yield None


def _primitive_storage_iter(view, child_factory):
    return iter(view.buffer(1).elements(view.offset, view.length))


def _nullable_primitive_storage_iter(view, child_factory):
    is_valid, data = view.buffers
    for is_valid, item in zip(
        is_valid.elements(view.offset, view.length),
        data.elements(view.offset, view.length),
    ):
        yield item if is_valid else None


def _array_view_nullable(array):
    return len(array.buffer(0)) != 0 and array.null_count != 0


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
