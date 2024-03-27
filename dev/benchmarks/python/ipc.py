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

import io
import os

import nanoarrow as na
from nanoarrow import ipc


class IpcReaderSuite:
    """
    Benchmarks for reading IPC streams
    """

    def setup(self):
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
        self.fixture_names = [
            "float64_basic.arrows",
            "float64_long.arrows",
            "float64_wide.arrows",
        ]
        self.fixture_buffer = {}
        for name in self.fixture_names:
            with open(self.fixture_path(name), "rb") as f:
                self.fixture_buffer[name] = f.read()

    def fixture_path(self, name):
        return os.path.join(self.fixtures_dir, name)

    def read_fixture_file(self, name):
        with ipc.Stream.from_path(self.fixture_path(name)) as in_stream:
            list(na.c_array_stream(in_stream))

    def read_fixture_buffer(self, name):
        f = io.BytesIO(self.fixture_buffer[name])
        with ipc.Stream.from_readable(f) as in_stream:
            list(na.c_array_stream(in_stream))

    def time_read_float64_basic_file(self):
        self.read_fixture_file("float64_basic.arrows")

    def time_read_float64_basic_buffer(self):
        self.read_fixture_buffer("float64_basic.arrows")

    def time_read_float64_long_buffer(self):
        self.read_fixture_buffer("float64_long.arrows")

    def time_read_float64_wide_buffer(self):
        self.read_fixture_buffer("float64_wide.arrows")
