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
from nanoarrow.c_schema import c_schema_view

import nanoarrow as na
from nanoarrow import extension


def test_basic_extension():
    class TestExtension(extension.Extension):
        def get_schema(self):
            return na.extension_type(na.int32(), "arrow.test")

        def get_params(self, c_schema):
            return {"parsed_key": "some parsed value"}

    instance = TestExtension()
    assert extension.register_extension(instance) is None

    # Check internal resolution
    assert extension.resolve_extension(c_schema_view(instance.get_schema())) is instance

    # Check Schema integration
    schema = na.extension_type(na.int32(), "arrow.test")
    assert schema.extension.parsed_key == "some parsed value"

    # Ensure other integrations fail if methods aren't implemented
    with pytest.raises(TypeError, match="get_iterable_appender"):
        assert na.Array([0], schema)

    with pytest.raises(TypeError, match="get_buffer_appender"):
        assert na.Array(bytearray([0]), schema)

    schema = na.extension_type(na.int32(), "arrow.test")
    storage_array = na.c_array([1, 2, 3], na.int32())
    _, storage_array_capsule = na.c_array(storage_array).__arrow_c_array__()
    array = na.Array(storage_array_capsule, schema)
    with pytest.raises(NotImplementedError, match="get_pyiter"):
        array.to_pylist()

    with pytest.raises(NotImplementedError, match="get_sequence_converter"):
        array.to_pysequence()

    other_instance = TestExtension()
    assert extension.register_extension(other_instance) is instance
    assert extension.unregister_extension("arrow.test") is other_instance
    with pytest.raises(KeyError):
        extension.unregister_extension("arrow.test")
