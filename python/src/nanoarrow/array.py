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

from functools import cached_property

from nanoarrow.schema import Schema
from nanoarrow._lib import CMaterializedArrayStream
from nanoarrow.c_lib import c_array_stream


class Array:

    def __init__(self, obj, schema=None) -> None:
        with c_array_stream(obj, schema=schema) as stream:
            self._data = CMaterializedArrayStream.from_c_array_stream(stream)

    @cached_property
    def schema(self) -> Schema:
        return Schema(self._data.schema)

    def __len__(self) -> int:
        return self._data.array_ends[len(self._data)]
