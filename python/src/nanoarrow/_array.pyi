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

import nanoarrow._device
from _typeshed import Incomplete
from nanoarrow._device import DeviceType as DeviceType
from typing import Iterable

DEVICE_CPU: nanoarrow._device.Device

class CArray:
    """Low-level ArrowArray wrapper

    This object is a literal wrapper around a read-only ArrowArray. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowArray structures are always released).

    See `nanoarrow.c_array()` for construction and usage examples.
    """

    buffers: Incomplete
    children: Incomplete
    device_id: Incomplete
    device_type: Incomplete
    device_type_id: Incomplete
    dictionary: Incomplete
    length: Incomplete
    n_buffers: Incomplete
    n_children: Incomplete
    null_count: Incomplete
    offset: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def allocate(schema) -> CArray:
        """Allocate a released ArrowArray"""

    def child(self, i): ...
    def is_valid(self) -> bool:
        """Check for a non-null and non-released underlying ArrowArray"""

    def view(self) -> CArrayView:
        """Allocate a :class:`CArrayView` to access the buffers of this array"""

    def __arrow_c_array__(self, requested_schema=...):
        """
        Get a pair of PyCapsules containing a C ArrowArray representation of the object.

        Parameters
        ----------
        requested_schema : PyCapsule | None
            A PyCapsule containing a C ArrowSchema representation of a requested
            schema. Not supported.

        Returns
        -------
        Tuple[PyCapsule, PyCapsule]
            A pair of PyCapsules containing a C ArrowSchema and ArrowArray,
            respectively.
        """

    def __getitem__(self, index):
        """Return self[key]."""

    def __len__(self) -> int:
        """Return len(self)."""

    def __reduce__(self): ...

class CArrayBuilder:
    """Helper for constructing an ArrowArray

    The primary function of this class is to wrap the nanoarrow C library calls
    that build up the components of an ArrowArray.
    """

    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def allocate(device=...):
        """Create a CArrayBuilder

        Allocates memory for an ArrowArray and populates it with nanoarrow's
        ArrowArray private_data/release callback implementation. This should
        usually be followed by :meth:`init_from_type` or :meth:`init_from_schema`.
        """

    def append_bytes(self, obj: Iterable[object]) -> CArrayBuilder: ...
    def append_strings(self, obj: Iterable[str | None]) -> CArrayBuilder: ...
    def ensure_buffers(self, n_buffers) -> CArrayBuilder: ...
    def finish(self, validation_level=...) -> CArray:
        """Finish building this array

        Performs any steps required to return a valid ArrowArray and optionally
        validates the output to ensure that the result is valid (given the information
        the array has available to it).

        Parameters
        ----------
        validation_level : None, "full", "default", "minimal", or "none", optional
            Explicitly define a validation level or use None to perform default
            validation if possible. Validation may not be possible if children
            were set that were not created by nanoarrow.
        """

    def finish_device(self):
        """Finish building this array and export to an ArrowDeviceArray

        Calls :meth:`finish`, propagating device information into an ArrowDeviceArray.
        """

    def init_from_schema(self, schema) -> CArrayBuilder: ...
    def init_from_type(self, type_id) -> CArrayBuilder: ...
    def is_empty(self) -> bool:
        """Check if any items have been appended to this builder"""

    def resolve_null_count(self) -> CArrayBuilder:
        """Ensure the output null count is synchronized with existing buffers

        Note that this will not attempt to access non-CPU buffers such that
        :attr:`null_count` might still be -1 after calling this method.
        """

    def set_buffer(self, i, buffer, move=...) -> CArrayBuilder:
        """Set an ArrowArray buffer

        Sets a buffer of this ArrowArray such the pointer at array->buffers[i] is
        equal to buffer->data and such that the buffer's lifcycle is managed by
        the array. If move is True, the input Python object that previously wrapped
        the ArrowBuffer will be invalidated, which is usually the desired behaviour
        if you built or imported a buffer specifically to build this array. If move
        is False (the default), this function will a make a shallow copy via another
        layer of Python object wrapping.
        """

    def set_child(self, i, c_array, move=...) -> CArrayBuilder:
        """Set an ArrowArray child

        Set a child of this array by performing a show copy or optionally
        transferring ownership to this object. The initialized child array
        must have been initialized before this call by initializing this
        builder with a schema containing the correct number of children.
        """

    def set_length(self, length) -> CArrayBuilder: ...
    def set_null_count(self, null_count) -> CArrayBuilder: ...
    def set_offset(self, offset) -> CArrayBuilder: ...
    def start_appending(self) -> CArrayBuilder:
        """Use append mode for building this ArrowArray

        Calling this method is required to produce a valid array prior to calling
        :meth:`append_strings` or `append_bytes`.
        """

    def __reduce__(self): ...

class CArrayView:
    """Low-level ArrowArrayView wrapper

    This object is a literal wrapper around an ArrowArrayView. It provides field accessors
    that return Python objects and handles the structure lifecycle (i.e., initialized
    ArrowArrayView structures are always released).

    See `nanoarrow.c_array_view()` for construction and usage examples.
    """

    buffers: Incomplete
    children: Incomplete
    dictionary: Incomplete
    layout: Incomplete
    length: Incomplete
    n_buffers: Incomplete
    n_children: Incomplete
    null_count: Incomplete
    offset: Incomplete
    storage_type: Incomplete
    storage_type_id: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def buffer(self, i): ...
    def buffer_type(self, i): ...
    def child(self, i): ...
    @staticmethod
    def from_array(array, device=...): ...
    @staticmethod
    def from_schema(schema): ...
    def __len__(self) -> int:
        """Return len(self)."""

    def __reduce__(self): ...

class CDeviceArray:
    """Low-level ArrowDeviceArray wrapper

    This object is a literal wrapper around an ArrowDeviceArray. It provides field accessors
    that return Python objects and handles the structure lifecycle (i.e., initialized
    ArrowDeviceArray structures are always released).

    See `nanoarrow.device.c_device_array()` for construction and usage examples.
    """

    array: Incomplete
    device_id: Incomplete
    device_type: Incomplete
    device_type_id: Incomplete
    schema: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def view(self) -> CArrayView: ...
    def __arrow_c_array__(self, requested_schema=...): ...
    def __arrow_c_device_array__(self, requested_schema=...): ...
    def __reduce__(self): ...
