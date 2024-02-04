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

import struct

import pytest
from nanoarrow.array_builder import c_array_from_pybuffer
from nanoarrow.c_lib import CBuffer

import nanoarrow as na


def test_c_array_from_pybuffer_uint8():
    data = b"abcdefg"
    c_array = c_array_from_pybuffer(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "uint8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_string():
    data = b"abcdefg"
    buffer = CBuffer().set_pybuffer(data).set_format("c")
    c_array = c_array_from_pybuffer(buffer.data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "int8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = CBuffer().set_pybuffer(packed).set_format("4s")

    c_array = c_array_from_pybuffer(buffer.data)
    assert c_array.length == len(items)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "fixed_size_binary"
    assert na.c_schema_view(c_array.schema).fixed_size == 4

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == items
