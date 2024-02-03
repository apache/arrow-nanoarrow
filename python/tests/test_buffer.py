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

from nanoarrow.c_lib import CBuffer


def test_buffer_invalid():
    invalid = CBuffer()

    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid._addr()
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.size_bytes
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.capacity_bytes
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.data

    assert repr(invalid) == "CBuffer(<invalid>)"


def test_buffer_empty():
    empty = CBuffer().set_empty()

    assert empty._addr() == 0
    assert empty.size_bytes == 0
    assert empty.capacity_bytes == 0
    assert bytes(empty.data) == b""

    assert repr(empty) == "CBuffer(binary[0 b] b'')"


def test_buffer_pybuffer():
    data = bytes(b"abcdefghijklmnopqrstuvwxyz")
    buffer = CBuffer().set_pybuffer(data)

    assert buffer.size_bytes == len(data)
    assert bytes(buffer.data) == b"abcdefghijklmnopqrstuvwxyz"

    assert repr(buffer).startswith("CBuffer(uint8[26 b] 97 98")
