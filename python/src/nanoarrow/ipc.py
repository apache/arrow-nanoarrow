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
    """Stream of serialized Arrow data

    Reads file paths or otherwise readable file objects that contain
    serialized Arrow data. Arrow documentation typically refers to this format
    as "Arrow IPC" because its origin was as a means to transmit tables between
    processes; however, this format can also be written to and read from files
    or URLs and is essentially a high-performance equivalent of a CSV file that
    does a better job maintaining type fidelity.

    Use :staticmethod:`from_readable`, :staticmethod:`from_path`, or
    :staticmethod:`from_url` to construct these streams.
    """

    def __init__(self):
        self._stream = None
        self._desc = None

    def _is_valid(self) -> bool:
        return self._stream is not None and self._stream.is_valid()

    def __arrow_c_stream__(self, requested_schema=None):
        """Export this stream as an ArrowArrayStream

        Implements the Arrow PyCapsule interface by transferring ownership of this
        input stream to an ArrowArrayStream wrapped by a PyCapsule.
        """
        if not self._is_valid():
            raise RuntimeError("nanoarrow.ipc.Stream is no longer valid")

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
        """Wrap an open readable object as an Arrow stream

        Wraps a readable object (specificially, an object that implements a
        ``readinto()`` method) as a non-owning Stream. Closing ``obj`` remains
        the caller's responsibility: neither this stream nor the resulting array
        stream will call ``obj.close()``.

        Parameters
        ----------
        obj : readable file-like
            An object implementing ``readinto()``.
        """
        out = Stream()
        out._stream = CIpcInputStream.from_readable(obj)
        out._desc = repr(obj)
        return out

    @staticmethod
    def from_path(obj, *args, **kwargs):
        """Wrap a local file as an Arrow stream

        Wraps a pathlike object (specificially, one that can be passed to ``open()``)
        as an owning Stream. The file will be opened in binary mode and will be closed
        when this stream or the resulting array stream is released.

        Parameters
        ----------
        obj : path-like
            A string or path-like object that can be passed to ``open()``
        """
        out = Stream()
        out._stream = CIpcInputStream.from_readable(
            open(obj, "rb", *args, **kwargs), close_stream=True
        )
        out._desc = repr(obj)
        return out

    @staticmethod
    def from_url(obj, *args, **kwargs):
        """Wrap a URL as an Arrow stream

        Wraps a URL (specificially, one that can be passed to
        ``urllib.request.urlopen()``) as an owning Stream. The URL will be
        closed when this stream or the resulting array stream is released.

        Parameters
        ----------
        obj : str
            A URL that can be passed to ``urllib.request.urlopen()``
        """
        import urllib.request

        out = Stream()
        out._stream = CIpcInputStream.from_readable(
            urllib.request.urlopen(obj, *args, **kwargs), close_stream=True
        )
        out._desc = repr(obj)
        return out

    @staticmethod
    def example():
        """Example Stream

        A self-contained example whose value is the serialized verison of
        ``DataFrame({"some_col": [1, 2, 3]})``. This may be used for testing
        and documentation and is useful because nanoarrow does not implement
        a writer to generate test data.
        """
        return Stream.from_readable(io.BytesIO(Stream.example_bytes()))

    @staticmethod
    def example_bytes():
        """Example stream bytes

        The underlying bytes of the :staticmethod:`example` Stream. This is useful
        for writing files or creating other types of test input.
        """
        return _EXAMPLE_IPC_SCHEMA + _EXAMPLE_IPC_BATCH

    def __repr__(self) -> str:
        if self._is_valid():
            return f"<nanoarrow.ipc.Stream {self._desc}>"
        else:
            return "<invalid nanoarrow.ipc.Stream>"


# A self-contained example whose value is the serialized verison of
# DataFrame({"some_col": [1, 2, 3]}). Used to make the tests self-contained
# since we don't have an IPC writer.
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
