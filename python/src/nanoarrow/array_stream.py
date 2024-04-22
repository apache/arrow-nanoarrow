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
    """High-level ArrayStream representation

    The ArrayStream is nanoarrow's high-level representation of zero
    or more contiguous arrays that have not neccessarily been materialized.
    This is in constrast to the nanoarrow :class:`Array`, which consists
    of zero or more contiguous arrays but is always fully-materialized.

    The :class:`ArrayStream` is similar to pyarrow's ``RecordBatchReader``
    except it can also represent streams of non-struct arrays. Its scope
    maps to that of an``ArrowArrayStream`` as represented by the Arrow C
    Stream interface.

    Parameters
    ----------
    obj : array or array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.ArrayStream([1, 2, 3], na.int32())
    <nanoarrow.ArrayStream: Schema(INT32)>
    """

    def __init__(self, obj, schema=None) -> None:
        self._c_array_stream = c_array_stream(obj, schema)

    @cached_property
    def schema(self):
        """The :class:`Schema` associated with this stream

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> stream.schema
        Schema(INT32)
        """
        return Schema(self._c_array_stream._get_cached_schema())

    def __arrow_c_stream__(self, requested_schema=None):
        return self._c_array_stream.__arrow_c_stream__(
            requested_schema=requested_schema
        )

    def __iter__(self) -> Iterable[Array]:
        for c_array in self._c_array_stream:
            yield Array(CMaterializedArrayStream.from_c_array(c_array))

    def read_all(self) -> Array:
        """Materialize the entire stream into an :class:`Array`

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> stream.read_all()
        nanoarrow.Array<int32>[3]
        1
        2
        3
        """
        return Array(self._c_array_stream)

    def read_next(self) -> Array:
        """Materialize the next contiguous :class:`Array` in this stream

        This method raises ``StopIteration`` if there are no more arrays
        in this stream.

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> stream.read_next()
        nanoarrow.Array<int32>[3]
        1
        2
        3
        """
        c_array = self._c_array_stream.get_next()
        return Array(CMaterializedArrayStream.from_c_array(c_array))

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self) -> None:
        """Release resources associated with this stream

        Note that it is usually preferred to use the context manager to ensure
        prompt release of resources (e.g., open files) associated with
        this stream.

        Examples
        --------
        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> with stream:
        ...     pass
        >>> stream.read_all()
        Traceback (most recent call last):
        ...
        RuntimeError: array stream is released

        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> stream.close()
        >>> stream.read_all()
        Traceback (most recent call last):
        ...
        RuntimeError: array stream is released
        """
        self._c_array_stream.release()

    def __repr__(self) -> str:
        cls = make_class_label(self, "nanoarrow")
        return f"<{cls}: {self.schema}>"

    @staticmethod
    def from_readable(obj):
        """Create an ArrayStream from an IPC stream in a readable file or buffer

        Examples
        --------
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import Stream
        >>> with na.ArrayStream.from_readable(Stream.example_bytes()) as stream:
        ...     stream.read_all()
        nanoarrow.Array<struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import Stream

        with Stream.from_readable(obj) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_path(obj, *args, **kwargs):
        """Create an ArrayStream from an IPC stream at a local file path

        Examples
        --------
        >>> import tempfile
        >>> import os
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import Stream
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with open(path, "wb") as f:
        ...         nbytes = f.write(Stream.example_bytes())
        ...
        ...     with na.ArrayStream.from_path(path) as stream:
        ...         stream.read_all()
        nanoarrow.Array<struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import Stream

        with Stream.from_path(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_url(obj, *args, **kwargs):
        """Create an ArrayStream from an IPC stream at a local file path

        Examples
        --------
        >>> import pathlib
        >>> import tempfile
        >>> import os
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import Stream
        >>> with tempfile.TemporaryDirectory() as td:
        ...     path = os.path.join(td, "test.arrows")
        ...     with open(path, "wb") as f:
        ...         nbytes = f.write(Stream.example_bytes())
        ...
        ...     uri = pathlib.Path(path).as_uri()
        ...     with na.ArrayStream.from_url(uri) as stream:
        ...         stream.read_all()
        nanoarrow.Array<struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import Stream

        with Stream.from_url(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)
