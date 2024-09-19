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

import _cython_3_0_11
from _typeshed import Incomplete

__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
init_array_stream: _cython_3_0_11.cython_function_or_method

class CIpcInputStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def from_readable(*args, **kwargs): ...
    def is_valid(self, *args, **kwargs): ...
    def release(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CIpcOutputStream:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def from_writable(*args, **kwargs): ...
    def is_valid(self, *args, **kwargs): ...
    def release(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CIpcWriter:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def is_valid(self, *args, **kwargs): ...
    def release(self, *args, **kwargs): ...
    def write_array_stream(self, *args, **kwargs): ...
    def write_array_view(self, *args, **kwargs): ...
    def write_end_of_stream(self, *args, **kwargs): ...
    def write_schema(self, *args, **kwargs): ...
    def __reduce__(self): ...

class PyStreamPrivate:
    close_obj: Incomplete
    obj: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def set_buffer(self, *args, **kwargs): ...
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""
