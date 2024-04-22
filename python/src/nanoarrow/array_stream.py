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

from nanoarrow._lib import CMaterializedArrayStream
from nanoarrow._repr_utils import make_class_label
from nanoarrow.array import Array
from nanoarrow.c_lib import c_array_stream
from nanoarrow.schema import Schema


class ArrayStream:
    def __init__(self, obj, schema=None) -> None:
        self._c_array_stream = c_array_stream(obj, schema)

    @cached_property
    def schema(self):
        return Schema(self._c_array_stream._get_cached_schema())

    def __arrow_c_stream__(self, requested_schema=None):
        return self._c_array_stream.__arrow_c_stream__(
            requested_schema=requested_schema
        )

    def __iter__(self) -> Iterable[Array]:
        for c_array in self._c_array_stream:
            yield Array(CMaterializedArrayStream.from_c_array(c_array))

    def read_all(self) -> Array:
        return Array(self._c_array_stream)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self) -> None:
        self._c_array_stream.release()

    def __repr__(self) -> str:
        cls = make_class_label(self, "nanoarrow")
        return f"<{cls}: {self.schema}>"

    @staticmethod
    def from_readable(obj):
        from nanoarrow.ipc import Stream

        with Stream.from_readable(obj) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_path(obj, *args, **kwargs):
        from nanoarrow.ipc import Stream

        with Stream.from_path(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_url(obj, *args, **kwargs):
        from nanoarrow.ipc import Stream

        with Stream.from_url(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)
