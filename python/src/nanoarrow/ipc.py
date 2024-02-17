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

from nanoarrow._ipc_lib import CIpcInputStream, init_array_stream
from nanoarrow._lib import CArrayStream


class IpcStream:
    def __init__(self, obj):
        if hasattr(obj, "readinto"):
            self._stream = CIpcInputStream.from_readable(obj)
        elif isinstance(obj, str):
            self._stream = CIpcInputStream.from_readable(
                open(obj, "rb"), close_stream=True
            )
        else:
            raise TypeError(f"Can't create IpcStream from object of type {type(obj).__name__}")

    def __arrow_c_stream__(self, requested_schema=None):
        array_stream = CArrayStream.allocate()
        init_array_stream(self._stream, array_stream._addr())
        return array_stream.__arrow_c_stream__(requested_schema=requested_schema)
