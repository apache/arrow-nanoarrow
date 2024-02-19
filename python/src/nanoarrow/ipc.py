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

from nanoarrow._ipc_lib import CIpcInputStream, init_array_stream
from nanoarrow._lib import CArrayStream


class Stream:
    """Stream of serialize Arrow data

    Reads file paths or otherwise readable file objects that contain
    serialized Arrow data. Arrow documentation typically refers to this format
    as "Arrow IPC" because its origin was as a means to transmit tables between
    processes; however, this format can also be written to and read from files
    or URLs and is essentially a high-performance equivalent of a CSV file that
    does a better job maintaining type fidelity.

    Use :staticmethod:`from_readable`, :staticmethod:`from_path`, or
    :staticmethod:`from_url`

    Parameters
    ----------
    obj : readable file or path-like
        A path to a file or

    """

    def __init__(self):
        self._stream = None

    def _is_valid(self):
        return self._stream is not None and self._stream.is_valid()

    def __arrow_c_stream__(self, requested_schema=None):
        array_stream = CArrayStream.allocate()
        init_array_stream(self._stream, array_stream._addr())
        return array_stream.__arrow_c_stream__(requested_schema=requested_schema)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._stream is not None:
            self._stream.release()

    @staticmethod
    def from_readable(obj):
        out = Stream()
        out._stream = CIpcInputStream.from_readable(obj)
        return out

    @staticmethod
    def from_path(obj, *args, **kwargs):
        out = Stream()
        out._stream = CIpcInputStream.from_readable(
            open(obj, "rb", *args, **kwargs), close_stream=True
        )
        return out

    @staticmethod
    def from_url(obj, *args, **kwargs):
        import urllib.request

        out = Stream()
        out._stream = CIpcInputStream.from_readable(
            urllib.request.urlopen(obj, *args, **kwargs), close_stream=True
        )
        return out

    @staticmethod
    def example():
        return Stream.from_readable(io.BytesIO(Stream.example_bytes()))

    @staticmethod
    def example_bytes():
        return _EXAMPLE_IPC_SCHEMA + _EXAMPLE_IPC_BATCH


# A self-contained example whose value is the serialized verison of
# DataFrame({"some_col": [1, 2, 3]})
_EXAMPLE_IPC_SCHEMA = (
    b"\xff\xff\xff\xff\x10\x01\x00\x00\x10\x00\x00\x00\x00\x00\x0a\x00\x0e\x00\x06"
    b"\x00\x05\x00\x08\x00\x0a\x00\x00\x00\x00\x01\x04\x00\x10\x00\x00\x00\x00\x00"
    b"\x0a\x00\x0c\x00\x00\x00\x04\x00\x08\x00\x0a\x00\x00\x00\x3c\x00\x00\x00\x04"
    b"\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00\x84\xff\xff\xff\x18\x00\x00\x00"
    b"\x04\x00\x00\x00\x0a\x00\x00\x00\x73\x6f\x6d\x65\x5f\x76\x61\x6c\x75\x65\x00"
    b"\x00\x08\x00\x00\x00\x73\x6f\x6d\x65\x5f\x6b\x65\x79\x00\x00\x00\x00\x01\x00"
    b"\x00\x00\x18\x00\x00\x00\x00\x00\x12\x00\x18\x00\x08\x00\x06\x00\x07\x00\x0c"
    b"\x00\x00\x00\x10\x00\x14\x00\x12\x00\x00\x00\x00\x00\x01\x02\x14\x00\x00\x00"
    b"\x70\x00\x00\x00\x08\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00"
    b"\x00\x73\x6f\x6d\x65\x5f\x63\x6f\x6c\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00"
    b"\x00\x00\x08\x00\x0c\x00\x04\x00\x08\x00\x08\x00\x00\x00\x20\x00\x00\x00\x04"
    b"\x00\x00\x00\x10\x00\x00\x00\x73\x6f\x6d\x65\x5f\x76\x61\x6c\x75\x65\x5f\x66"
    b"\x69\x65\x6c\x64\x00\x00\x00\x00\x0e\x00\x00\x00\x73\x6f\x6d\x65\x5f\x6b\x65"
    b"\x79\x5f\x66\x69\x65\x6c\x64\x00\x00\x08\x00\x0c\x00\x08\x00\x07\x00\x08\x00"
    b"\x00\x00\x00\x00\x00\x01\x20\x00\x00\x00\x00\x00\x00\x00"
)

_EXAMPLE_IPC_BATCH = (
    b"\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16"
    b"\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00"
    b"\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0a\x00\x18\x00\x0c\x00\x04"
    b"\x00\x08\x00\x0a\x00\x00\x00\x3c\x00\x00\x00\x10\x00\x00\x00\x03\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00"
    b"\x03\x00\x00\x00\x00\x00\x00\x00"
)
