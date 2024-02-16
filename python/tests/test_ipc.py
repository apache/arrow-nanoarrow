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

import os

import pytest
from nanoarrow.ipc import IpcStream

import nanoarrow as na

os.environ[
    "NANOARROW_ARROW_TESTING_DIR"
] = "/Users/deweydunnington/Desktop/rscratch/arrow-testing"


def get_test_ipc_filename(name):
    testing_dir = os.environ.get("NANOARROW_ARROW_TESTING_DIR")
    if not testing_dir:
        pytest.skip("NANOARROW_ARROW_TESTING_DIR not set")

    return os.path.join(
        testing_dir,
        "data",
        "arrow-ipc-stream",
        "integration",
        "1.0.0-littleendian",
        f"generated_{name}.stream",
    )


def test_ipc_from_readable():
    filename = get_test_ipc_filename("null")
    input = IpcStream(open(filename, "rb"))
    stream = na.c_array_stream(input)
    schema = stream.get_schema()
    assert schema.format == "+s"
    for array in stream:
        assert array.n_buffers == 1
