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

import nanoarrow as na
from nanoarrow import ipc


class IpcReaderSuite:
    """
    Benchmarks for reading IPC streams
    """

    def setup(self):
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")

    def fixture_path(self, name):
        return os.path.join(self.fixtures_dir, name)

    def time_read_float64_basic(self):
        na.Array(ipc.Stream.from_path(self.fixture_path("float64_basic.arrows")))

    def time_read_float64_long(self):
        na.Array(ipc.Stream.from_path(self.fixture_path("float64_long.arrows")))

    def time_read_float64_wide(self):
        na.Array(ipc.Stream.from_path(self.fixture_path("float64_wide.arrows")))
