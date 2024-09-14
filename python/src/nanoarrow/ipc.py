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

from nanoarrow._array_stream import CArrayStream
from nanoarrow._ipc_lib import (
    CIpcInputStream,
    CIpcOutputStream,
    CIpcWriter,
    init_array_stream,
)
from nanoarrow._utils import obj_is_buffer
from nanoarrow.array import c_array
from nanoarrow.array_stream import c_array_stream
from nanoarrow.iterator import ArrayViewBaseIterator

from nanoarrow import _repr_utils


class InputStream:
    """Stream of serialized Arrow data

    Reads file paths or otherwise readable file objects that contain
    serialized Arrow data. Arrow documentation typically refers to this format
    as "Arrow IPC" because its origin was as a means to transmit tables between
    processes; however, this format can also be written to and read from files
    or URLs and is essentially a high-performance equivalent of a CSV file that
    does a better job maintaining type fidelity.

    Use :staticmethod:`from_readable`, :staticmethod:`from_path`, or
    :staticmethod:`from_url` to construct these streams.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow.ipc import InputStream
    >>> with InputStream.example() as inp, na.c_array_stream(inp) as stream:
    ...     stream
    <nanoarrow.c_array_stream.CArrayStream>
    - get_schema(): struct<some_col: int32>
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
            raise RuntimeError("nanoarrow.ipc.InputStream is no longer valid")

        with CArrayStream.allocate() as array_stream:
            init_array_stream(self._stream, array_stream._addr())
            array_stream._get_cached_schema()
            return array_stream.__arrow_c_stream__(requested_schema=requested_schema)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._stream is not None:
            self._stream.release()

    @staticmethod
    def from_readable(obj):
        """Wrap an open readable file or buffer as an Arrow IPC stream

        Wraps a readable object (specificially, an object that implements a
        ``readinto()`` method) as a non-owning InputStream. Closing ``obj`` remains
        the caller's responsibility: neither this stream nor the resulting array
        stream will call ``obj.close()``.

        Parameters
        ----------
        obj : readable file-like or buffer
            An object implementing the Python buffer protocol or ``readinto()``.

        Examples
        --------

        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import InputStream
        >>> ipc_stream = InputStream.from_readable(InputStream.example_bytes())
        >>> na.c_array_stream(ipc_stream)
        <nanoarrow.c_array_stream.CArrayStream>
        - get_schema(): struct<some_col: int32>
        """
        if not hasattr(obj, "readinto") and obj_is_buffer(obj):
            close_obj = True
            obj = io.BytesIO(obj)
        else:
            close_obj = False

        out = InputStream()
        out._stream = CIpcInputStream.from_readable(obj, close_obj=close_obj)
        out._desc = repr(obj)
        return out

    @staticmethod
    def from_path(obj, *args, **kwargs):
        """Wrap a local file as an IPC stream

        Wraps a pathlike object (specificially, one that can be passed to ``open()``)
        as an owning InputStream. The file will be opened in binary mode and will be
        closed when this stream or the resulting array stream is released.

        Parameters
        ----------
        obj : path-like
            A string or path-like object that can be passed to ``open()``

        Examples
        --------

        >>> import tempfile
        >>> import os
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import InputStream
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with open(path, "wb") as f:
        ...         nbytes = f.write(InputStream.example_bytes())
        ...
        ...     with InputStream.from_path(path) as inp, na.c_array_stream(inp) as s:
        ...         s
        <nanoarrow.c_array_stream.CArrayStream>
        - get_schema(): struct<some_col: int32>
        """
        out = InputStream()
        out._stream = CIpcInputStream.from_readable(
            open(obj, "rb", *args, **kwargs), close_obj=True
        )
        out._desc = repr(obj)
        return out

    @staticmethod
    def from_url(obj, *args, **kwargs):
        """Wrap a URL as an IPC stream

        Wraps a URL (specificially, one that can be passed to
        ``urllib.request.urlopen()``) as an owning InputStream. The URL will be
        closed when this stream or the resulting array stream is released.

        Parameters
        ----------
        obj : str
            A URL that can be passed to ``urllib.request.urlopen()``

        Examples
        --------

        >>> import pathlib
        >>> import tempfile
        >>> import os
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import InputStream
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with open(path, "wb") as f:
        ...         nbytes = f.write(InputStream.example_bytes())
        ...
        ...     uri = pathlib.Path(path).as_uri()
        ...     with InputStream.from_url(uri) as inp, na.c_array_stream(inp) as stream:
        ...         stream
        <nanoarrow.c_array_stream.CArrayStream>
        - get_schema(): struct<some_col: int32>
        """
        import urllib.request

        out = InputStream()
        out._stream = CIpcInputStream.from_readable(
            urllib.request.urlopen(obj, *args, **kwargs), close_obj=True
        )
        out._desc = repr(obj)
        return out

    @staticmethod
    def example():
        """Example IPC InputStream

        A self-contained example whose value is the serialized version of
        ``DataFrame({"some_col": [1, 2, 3]})``. This may be used for testing
        and documentation and is useful because nanoarrow does not implement
        a writer to generate test data.

        Examples
        --------

        >>> from nanoarrow.ipc import InputStream
        >>> InputStream.example()
        <nanoarrow.ipc.InputStream <_io.BytesIO object at ...>>
        """
        return InputStream.from_readable(InputStream.example_bytes())

    @staticmethod
    def example_bytes():
        """Example stream bytes

        The underlying bytes of the :staticmethod:`example` InputStream. This is useful
        for writing files or creating other types of test input.

        Examples
        --------

        >>> import os
        >>> import tempfile
        >>> from nanoarrow.ipc import InputStream
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with open(path, "wb") as f:
        ...         f.write(InputStream.example_bytes())
        440
        """
        return _EXAMPLE_IPC_SCHEMA + _EXAMPLE_IPC_BATCH

    def __repr__(self) -> str:
        class_label = _repr_utils.make_class_label(self)
        if self._is_valid():
            return f"<{class_label} {self._desc}>"
        else:
            return f"<{class_label} <invalid>>"


class StreamWriter:
    """Write streams of serialized Arrow data

    Provides various ways of writing Arrow schemas and record batches as
    binary data serialized using the Arrow IPC streaming format.

    Use :staticmethod:`from_writeable` or :staticmethod:`from_path`, or
    to construct a writer.

    Examples
    --------

    >>> import io
    >>> import nanoarrow as na
    >>> from nanoarrow.ipc import StreamWriter
    >>>
    >>> out = io.BytesIO()
    >>> array = na.c_array_from_buffers(
    ...     na.struct({"some_col": na.int32()}),
    ...     length=3,
    ...     buffers=[],
    ...     children=[na.c_array([1, 2, 3], na.int32())]
    ... )
    >>>
    >>> with StreamWriter.from_writable(out) as writer:
    ...     writer.write_stream(array)
    >>>
    >>> na.ArrayStream.from_readable(out.getvalue()).read_all()
    nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
    {'some_col': 1}
    {'some_col': 2}
    {'some_col': 3}
    """

    def __init__(self):
        self._writer = None
        self._desc = None
        self._iterator = None

    def _is_valid(self) -> bool:
        return self._writer is not None and self._writer.is_valid()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def release(self):
        """Close stream without writing the end-of-stream marker"""
        if not self._is_valid():
            return

        self._writer.release()
        self._writer = None

    def close(self):
        """Close stream and write end-of-stream marker"""
        if not self._is_valid():
            return

        self._writer.write_end_of_stream()
        self.release()

    def write_array(self, obj, schema=None, *, write_schema=None):
        """Interpret obj as an array and write to stream

        Parameters
        ----------
        obj : array-like
            An array-like object as sanitized by :func:`c_array`.
        schema : schema-like, optional
            An optional schema, passed to :func:`c_array`.
        write_schema : bool, optional
            See :meth:`write_stream`.
        """
        obj = c_array(obj)
        return self.write_stream(obj, schema, write_schema=write_schema)

    def write_stream(self, obj, schema=None, *, write_schema=None):
        """Interpret obj as a stream of arrays and write to stream

        Writes all arrays from obj to the output stream.

        Parameters
        ----------
        obj : array stream-like
            An array-like or array stream-like object as sanitized by
            :func:`c_array_stream`.
        schema : schema-like, optional
            An optional schema, passed to :func:`c_array_stream`.
        write_schema : bool, optional
            If True, the schema will always be written to the output stream; if False,
            the schema will never be written to the output stream. If omitted, the
            schema will be written if nothing has yet been written to the output.
        """
        if not self._is_valid():
            raise ValueError("Can't write to released StreamWriter")

        with c_array_stream(obj, schema=schema) as stream:
            if self._iterator is None:
                self._iterator = ArrayViewBaseIterator(stream._get_cached_schema())
                if write_schema is None:
                    write_schema = True

            if write_schema:
                self._writer.write_schema(self._iterator._schema)

            for array in stream:
                self._iterator._set_array(array)
                self._writer.write_array_view(self._iterator._array_view)

    def serialize_stream(self, obj, schema=None):
        """Interpret obj as a stream of arrays, write to stream, and close

        Like :meth:`write_stream` except always writes a schema message and
        always appends the end-of-stream marker to the output. This method
        also takes a potentially more efficient path that uses fewer Python
        calls at the expense of less flexibility. After calling this method,
        the writer is released and subequent calls to methods will error.

        Parameters
        ----------
        obj : array stream-like
            An array-like or array stream-like object as sanitized by
            :func:`c_array_stream`.
        schema : schema-like, optional
            An optional schema, passed to :func:`c_array_stream`.
        """
        if not self._is_valid():
            raise ValueError("Can't write to released StreamWriter")

        # If we've already written a schema or we've explicitly been asked
        # to write one, we can't write using the stream writer because it
        # automatically appends a schema. We can, however, write the rest
        # of the stream and EOS using write().
        if self._iterator is not None:
            raise ValueError("Can't serialize_stream() into a non-empty writer")

        # Write the entire stream and release the writer. We can't
        # use close() because that would write the EOS and the stream
        # writer has already appended this.
        with self, c_array_stream(obj, schema=schema) as stream:
            self._writer.write_array_stream(stream._addr())
            self.release()

    @staticmethod
    def from_writable(obj):
        """Write an Arrow IPC stream to a writable file

        Wraps a writable object (specificially, an object that implements a
        ``write()`` method) as a non-owning StreamWriter. Closing ``obj`` remains
        the caller's responsibility (i.e., closing this object will not call
        ``obj.close()``.

        Parameters
        ----------
        obj : A writable file-like object supporting ``write()``.

        Examples
        --------

        >>> import io
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import StreamWriter
        >>>
        >>> out = io.BytesIO()
        >>> array = na.c_array_from_buffers(
        ...     na.struct({"some_col": na.int32()}),
        ...     length=3,
        ...     buffers=[],
        ...     children=[na.c_array([1, 2, 3], na.int32())]
        ... )
        >>>
        >>> with StreamWriter.from_writable(out) as writer:
        ...     writer.write_stream(array)
        >>>
        >>> na.ArrayStream.from_readable(out.getvalue()).read_all()
        nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        out = StreamWriter()
        stream = CIpcOutputStream.from_writable(obj, close_obj=False)
        out._desc = repr(obj)

        out._writer = CIpcWriter(stream)
        return out

    @staticmethod
    def from_path(obj, *args, **kwargs):
        """Wrap a local file as an IPC stream

        Wraps a pathlike object (specificially, one that can be passed to ``open()``)
        as an owning StreamWriter. The file will be opened in (writable) binary mode and
        will be closed when the returned writer is closed.

        Parameters
        ----------
        obj : path-like
            A string or path-like object that can be passed to ``open()``

        Examples
        --------
        >>> import os
        >>> import tempfile
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import StreamWriter
        >>>
        >>> array = na.c_array_from_buffers(
        ...     na.struct({"some_col": na.int32()}),
        ...     length=3,
        ...     buffers=[],
        ...     children=[na.c_array([1, 2, 3], na.int32())]
        ... )
        >>>
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with StreamWriter.from_path(path) as writer:
        ...         writer.write_stream(array)
        ...
        ...     with na.ArrayStream.from_path(path) as stream:
        ...         stream.read_all()
        nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        out = StreamWriter()
        stream = CIpcOutputStream.from_writable(
            open(obj, "wb", *args, **kwargs), close_obj=True
        )
        out._writer = CIpcWriter(stream)
        return out


# A self-contained example whose value is the serialized verison of
# DataFrame({"some_col": [1, 2, 3]}). Used to make the tests and documentation
# self-contained.
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
