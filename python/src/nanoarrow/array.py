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
from typing import Iterable

from nanoarrow._lib import CArray, CMaterializedArrayStream
from nanoarrow.c_lib import c_array, c_array_stream
from nanoarrow.schema import Schema


class Array:
    def __init__(self, obj, schema=None) -> None:
        if isinstance(obj, Array) and schema is None:
            self._data = obj._data
            return

        if isinstance(obj, CArray) and schema is None:
            self._data = CMaterializedArrayStream.from_c_array(obj)
            return

        with c_array_stream(obj, schema=schema) as stream:
            self._data = CMaterializedArrayStream.from_c_array_stream(stream)

    def __arrow_c_stream__(self, requested_schema=None):
        return self._data.__arrow_c_stream__(requested_schema=requested_schema)

    def __arrow_c_array__(self, requested_schema=None):
        if len(self._data) == 0:
            return c_array([], schema=self._data.schema).__arrow_c_array__(
                requested_schema=requested_schema
            )
        elif len(self._data) == 1:
            return self._data[0].__arrow_c_array__(requested_schema=requested_schema)

        raise ValueError(
            f"Can't export Array with {len(self._data)} chunks to ArrowArray"
        )

    @cached_property
    def schema(self) -> Schema:
        return Schema(self._data.schema)

    @property
    def n_chunks(self) -> int:
        return len(self._data)

    @property
    def chunks(self) -> Iterable:
        for array in self._data:
            yield Array(array)

    def chunk(self, i):
        return Array(self._data[i])

    def __len__(self) -> int:
        return self._data.array_ends[len(self._data)]
