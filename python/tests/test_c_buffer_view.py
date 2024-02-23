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

import pytest
from nanoarrow.c_lib import c_array_view

import nanoarrow as na


def test_buffer_view_bool():
    bool_array_view = c_array_view([1, 0, 0, 1], na.bool())
    view = bool_array_view.buffer(1)

    assert view.element_size_bits == 1
    assert view.size_bytes == 1
    assert view.data_type_id == na.Type.BOOL.value
    assert view.data_type == "bool"
    assert view.format == "B"

    # Check item interface
    assert len(view) == 1
    assert view[0] == 0b1001
    assert list(view) == [0b1001]

    # Check against buffer protocol
    mv = memoryview(view)
    assert len(mv) == len(view)
    assert mv[0] == view[0]
    assert list(mv) == list(view)

    # Check element interface
    assert view.n_elements == 8
    assert list(view.elements()) == [True, False, False, True] + [False] * 4
    assert [view.element(i) for i in range(8)] == list(view.elements())

    # Check element slices
    assert list(view.elements(0, 4)) == [True, False, False, True]
    assert list(view.elements(1, 3)) == [False, False, True]

    msg = "do not describe a valid slice"
    with pytest.raises(IndexError, match=msg):
        view.elements(-1, None)
    with pytest.raises(IndexError, match=msg):
        view.elements(0, -1)
    with pytest.raises(IndexError, match=msg):
        view.elements(0, 9)

    # Check repr
    assert "10010000" in repr(view)


def test_buffer_view_non_bool():
    array_view = c_array_view([1, 2, 3, 5], na.int32())
    view = array_view.buffer(1)

    assert view.element_size_bits == 32
    assert view.size_bytes == 4 * 4
    assert view.data_type_id == na.Type.INT32.value
    assert view.data_type == "int32"
    assert view.format == "i"

    # Check item interface
    assert len(view) == 4
    assert list(view) == [1, 2, 3, 5]
    assert [view[i] for i in range(4)] == list(view)

    # Check against buffer protocol
    mv = memoryview(view)
    assert len(mv) == len(view)
    assert mv[0] == view[0]
    assert [mv[i] for i in range(4)] == [view[i] for i in range(4)]

    # Check element interface
    assert view.n_elements == len(view)
    assert list(view.elements()) == list(view)
    assert [view.element(i) for i in range(4)] == list(view.elements())

    # Check element slices
    assert list(view.elements(0, 3)) == [1, 2, 3]
    assert list(view.elements(1, 3)) == [2, 3, 5]

    with pytest.raises(IndexError, match="do not describe a valid slice"):
        view.elements(-1, None)
    with pytest.raises(IndexError, match="do not describe a valid slice"):
        view.elements(0, -1)
    with pytest.raises(IndexError, match="do not describe a valid slice"):
        view.elements(1, 4)

    # Check repr
    assert "1 2 3 5" in repr(view)
