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

import nanoarrow as na


def test_extension_bool8():
    schema = na.bool8()
    assert schema.type == na.Type.EXTENSION
    assert schema.extension.storage.type == na.Type.INT8
    assert schema.extension.name == "arrow.bool8"
    assert schema.extension.metadata == b""

    assert na.bool8(nullable=False).nullable is False

    bool8_array = na.Array([True, False, True, True], na.bool8())
    assert bool8_array.schema.type == na.Type.EXTENSION
    assert bool8_array.schema.extension.name == "arrow.bool8"
    assert bool8_array.to_pylist() == [True, False, True, True]

    sequence = bool8_array.to_pysequence()
    assert list(sequence) == [True, False, True, True]

    bool8_array = na.Array([True, False, None, True], na.bool8())
    assert bool8_array.to_pylist() == [True, False, None, True]

    sequence = bool8_array.to_pysequence(handle_nulls=na.nulls_separate())
    assert list(sequence[1]) == [True, False, False, True]
    assert list(sequence[0]) == [True, True, False, True]

    bool8_array = na.Array(sequence[1], na.bool8())
    assert bool8_array.to_pylist() == [True, False, False, True]
