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

from _typeshed import Incomplete

class CIpcInputStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def from_readable(obj, close_obj=...): ...
    def is_valid(self): ...
    def release(self): ...
    def __reduce__(self): ...

class CIpcOutputStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def from_writable(obj, close_obj=...): ...
    def is_valid(self): ...
    def release(self): ...
    def __reduce__(self): ...

class CIpcWriter:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def is_valid(self): ...
    def release(self): ...
    def write_array_stream(self, stream_addr): ...
    def write_array_view(self, array_view): ...
    def write_end_of_stream(self): ...
    def write_schema(self, schema): ...
    def __reduce__(self): ...

class PyStreamPrivate:
    close_obj: Incomplete
    obj: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def set_buffer(self, addr, size_bytes): ...
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

def init_array_stream(input_stream, out): ...
