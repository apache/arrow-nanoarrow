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
from nanoarrow._array import CArray
from nanoarrow._schema import CSchema
from typing import Iterable

class CArrayStream:
    """Low-level ArrowArrayStream wrapper

    This object is a literal wrapper around an ArrowArrayStream. It provides methods that
    that wrap the underlying C callbacks and handles the C Data interface lifecycle
    (i.e., initialized ArrowArrayStream structures are always released).

    See `nanoarrow.c_array_stream()` for construction and usage examples.
    """

    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def allocate() -> CArrayStream:
        """Allocate a released ArrowArrayStream"""

    @staticmethod
    def from_c_arrays(
        arrays: list[CArray], schema, move=..., validate=...
    ) -> CArrayStream:
        """Create an ArrowArrayStream from an existing set of arrays

        Given a previously resolved list of arrays, create an ArrowArrayStream
        representation of the sequence of chunks.

        Parameters
        ----------
        arrays : List[CArray]
            A list of arrays to use as batches.
        schema : CSchema
            The schema that will be returned. Must be type equal with the schema
            of each array (this is checked if validate is ``True``)
        move : bool, optional
            If True, transfer ownership from each array instead of creating a
            shallow copy. This is only safe if the caller knows the origin of the
            arrays and knows that they will not be accessed after this stream has been
            created.
        validate : bool, optional
            If True, enforce type equality between the provided schema and the schema
            of each array.
        """

    def get_next(self) -> CArray:
        """Get the next Array from this stream

        Raises StopIteration when there are no more arrays in this stream.
        """

    def get_schema(self) -> CSchema:
        """Get the schema associated with this stream

        Calling this method will always issue a call to the underlying stream's
        get_schema callback.
        """

    def is_valid(self) -> bool:
        """Check for a non-null and non-released underlying ArrowArrayStream"""

    def release(self):
        """Explicitly call the release callback of this stream"""

    def __arrow_c_stream__(self, requested_schema=...):
        """
        Export the stream as an Arrow C stream PyCapsule.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        PyCapsule
        """

    def __enter__(self): ...
    def __exit__(self, *args, **kwargs): ...
    def __iter__(self):
        """Implement iter(self)."""

    def __next__(self): ...
    def __reduce__(self): ...

class CMaterializedArrayStream:
    """Optimized representation of a fully consumed ArrowArrayStream

    This class provides a data structure similar to pyarrow's ChunkedArray
    where each consumed array is a referenced-counted shared array. This
    class wraps the utilities provided by the nanoarrow C library to iterate
    over and facilitate log(n) random access to items in this container.
    """

    arrays: Incomplete
    n_arrays: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def array(self, i) -> CArray: ...
    def child(self, i) -> CMaterializedArrayStream: ...
    @staticmethod
    def from_c_array(array) -> CMaterializedArrayStream:
        """ "Create a materialized array stream from a single array"""

    @staticmethod
    def from_c_array_stream(stream) -> CMaterializedArrayStream:
        """ "Create a materialized array stream from an unmaterialized ArrowArrayStream"""

    @staticmethod
    def from_c_arrays(arrays: Iterable[CArray], schema, validate=...):
        """ "Create a materialized array stream from an existing iterable of arrays

        This is slightly more efficient than creating a stream and then consuming it
        because the implementation can avoid a shallow copy of each array.
        """

    def __arrow_c_stream__(self, requested_schema=...): ...
    def __getitem__(self, index):
        """Return self[key]."""

    def __iter__(self):
        """Implement iter(self)."""

    def __len__(self) -> int:
        """Return len(self)."""

    def __reduce__(self): ...
