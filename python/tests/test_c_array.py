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

import nanoarrow as na


def test_c_array_from_c_array():
    c_array = na.c_array([1, 2, 3], na.int32())
    c_array_from_c_array = na.c_array(c_array)
    assert c_array_from_c_array.length == c_array.length
    assert c_array_from_c_array.buffers == c_array.buffers

    assert list(c_array.view().buffer(1)) == [1, 2, 3]


def test_c_array_from_capsule_protocol():
    class CArrayWrapper:
        def __init__(self, obj):
            self.obj = obj

        def __arrow_c_array__(self, *args, **kwargs):
            return self.obj.__arrow_c_array__(*args, **kwargs)

    c_array = na.c_array([1, 2, 3], na.int32())
    c_array_wrapper = CArrayWrapper(c_array)
    c_array_from_protocol = na.c_array(c_array_wrapper)
    assert c_array_from_protocol.length == c_array.length
    assert c_array_from_protocol.buffers == c_array.buffers

    assert list(c_array_from_protocol.view().buffer(1)) == [1, 2, 3]


def test_c_array_from_old_pyarrow():
    # Simulate a pyarrow Array with no __arrow_c_array__
    class MockLegacyPyarrowArray:
        def __init__(self, obj):
            self.obj = obj

        def _export_to_c(self, *args):
            return self.obj._export_to_c(*args)

    MockLegacyPyarrowArray.__module__ = "pyarrow.lib"

    pa = pytest.importorskip("pyarrow")
    array = MockLegacyPyarrowArray(pa.array([1, 2, 3], pa.int32()))

    c_array = na.c_array(array)
    assert c_array.length == 3
    assert c_array.schema.format == "i"

    assert list(c_array.view().buffer(1)) == [1, 2, 3]

    # Make sure that this heuristic won't result in trying to import
    # something else that has an _export_to_c method
    with pytest.raises(TypeError, match="Can't resolve ArrayBuilder"):
        not_array = pa.int32()
        assert hasattr(not_array, "_export_to_c")
        na.c_array(not_array)


def test_c_array_from_bare_capsule():
    c_array = na.c_array([1, 2, 3], na.int32())

    # Check from bare capsule without supplying a schema
    schema_capsule, array_capsule = c_array.__arrow_c_array__()
    del schema_capsule
    c_array_from_capsule = na.c_array(array_capsule)
    assert c_array_from_capsule.length == c_array.length
    assert c_array_from_capsule.buffers == c_array.buffers

    # Check from bare capsule supplying a schema
    schema_capsule, array_capsule = c_array.__arrow_c_array__()
    c_array_from_capsule = na.c_array(array_capsule, schema_capsule)
    assert c_array_from_capsule.length == c_array.length
    assert c_array_from_capsule.buffers == c_array.buffers

    assert list(c_array_from_capsule.view().buffer(1)) == [1, 2, 3]


def test_c_array_type_not_supported():
    msg = "Can't resolve ArrayBuilder for object of type NoneType"
    with pytest.raises(TypeError, match=msg):
        na.c_array(None)


def test_c_array_slice():
    array = na.c_array([1, 2, 3], na.int32())
    assert array.offset == 0
    assert array.length == 3

    array2 = array[:]
    assert array.offset == 0
    assert array.length == 3
    assert array.buffers == array2.buffers

    array2 = array[:2]
    assert array2.offset == 0
    assert array2.length == 2

    array2 = array[:-1]
    assert array2.offset == 0
    assert array2.length == 2

    array2 = array[1:]
    assert array2.offset == 1
    assert array2.length == 2

    array2 = array[-2:]
    assert array2.offset == 1
    assert array2.length == 2


def test_c_array_slice_errors():
    array = na.c_array([1, 2, 3], na.int32())

    with pytest.raises(TypeError):
        array[None]
    with pytest.raises(IndexError):
        array[4:]
    with pytest.raises(IndexError):
        array[:4]
    with pytest.raises(IndexError):
        array[1:0]
