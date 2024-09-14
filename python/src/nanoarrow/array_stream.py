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
from typing import Iterable, Tuple

from nanoarrow._array_stream import CMaterializedArrayStream
from nanoarrow._repr_utils import make_class_label
from nanoarrow.array import Array
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.iterator import iter_py, iter_tuples
from nanoarrow.schema import Schema, _schema_repr
from nanoarrow.visitor import ArrayViewVisitable


class ArrayStream(ArrayViewVisitable):
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
    nanoarrow.ArrayStream<int32>
    """

    def __init__(self, obj, schema=None) -> None:
        self._c_array_stream = c_array_stream(obj, schema)

    @cached_property
    def schema(self):
        """The :class:`Schema` associated with this stream

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> stream.schema
        <Schema> int32
        """
        return Schema(self._c_array_stream._get_cached_schema())

    def __arrow_c_stream__(self, requested_schema=None):
        return self._c_array_stream.__arrow_c_stream__(
            requested_schema=requested_schema
        )

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

    def __iter__(self) -> Iterable[Array]:
        for c_array in self._c_array_stream:
            yield Array(CMaterializedArrayStream.from_c_array(c_array))

    def iter_chunks(self) -> Iterable[Array]:
        """Iterate over contiguous Arrays in this stream

        For the :class:`ArrayStream`, this is the same as iterating over
        the stream itself.

        Examples
        --------

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> for chunk in stream:
        ...     print(chunk)
        nanoarrow.Array<int32>[3]
        1
        2
        3
        """
        return iter(self)

    def iter_py(self) -> Iterable:
        """Iterate over the default Python representation of each element.

        Examples
        --------

        >>> import nanoarrow as na
        >>> stream = na.ArrayStream([1, 2, 3], na.int32())
        >>> for item in stream.iter_py():
        ...     print(item)
        1
        2
        3
        """
        return iter_py(self)

    def iter_tuples(self) -> Iterable[Tuple]:
        """Iterate over rows of a struct stream as tuples

        Examples
        --------

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> batch = pa.record_batch(
        ...     [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])],
        ...     names=["col1", "col2"]
        ... )
        >>> stream = na.ArrayStream(batch)
        >>> for item in stream.iter_tuples():
        ...     print(item)
        (1, 'a')
        (2, 'b')
        (3, 'c')
        """
        return iter_tuples(self)

    def __repr__(self) -> str:
        cls = make_class_label(self, "nanoarrow")
        schema_repr = _schema_repr(self.schema, prefix="", include_metadata=False)
        return f"{cls}<{schema_repr}>"

    @staticmethod
    def from_readable(obj):
        """Create an ArrayStream from an IPC stream in a readable file or buffer

        Examples
        --------
        >>> import nanoarrow as na
        >>> from nanoarrow.ipc import InputStream
        >>> with na.ArrayStream.from_readable(InputStream.example_bytes()) as stream:
        ...     stream.read_all()
        nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import InputStream

        with InputStream.from_readable(obj) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_path(obj, *args, **kwargs):
        """Create an ArrayStream from an IPC stream at a local file path

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
        ...     with na.ArrayStream.from_path(path) as stream:
        ...         stream.read_all()
        nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import InputStream

        with InputStream.from_path(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)

    @staticmethod
    def from_url(obj, *args, **kwargs):
        """Create an ArrayStream from an IPC stream at a URL

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
        ...     with na.ArrayStream.from_url(uri) as stream:
        ...         stream.read_all()
        nanoarrow.Array<non-nullable struct<some_col: int32>>[3]
        {'some_col': 1}
        {'some_col': 2}
        {'some_col': 3}
        """
        from nanoarrow.ipc import InputStream

        with InputStream.from_url(obj, *args, **kwargs) as ipc_stream:
            return ArrayStream(ipc_stream)
