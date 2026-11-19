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

DEVICE_CPU: nanoarrow._device.Device

class CBuffer:
    """Wrapper around readable owned buffer content

    Like the CBufferView, the CBuffer represents readable buffer content; however,
    unlike the CBufferView, the CBuffer always represents a valid ArrowBuffer C object.
    Whereas the CBufferView is primarily concerned with accessing the contents of a
    buffer, the CBuffer is primarily concerned with managing ownership of an external
    buffer such that it exported as an Arrow array.
    """

    data_type: Incomplete
    data_type_id: Incomplete
    device: Incomplete
    element_size_bits: Incomplete
    format: Incomplete
    itemsize: Incomplete
    n_elements: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def element(self, i): ...
    def elements(self, offset=..., length=...): ...
    @staticmethod
    def empty():
        """Create an empty CBuffer"""
    @staticmethod
    def from_dlpack(obj, stream=...) -> CBuffer:
        """Create a CBuffer using the DLPack protocol

        Wraps a tensor from an external library as a CBuffer that can be used
        to create an array.

        Parameters
        ----------
        obj : object with a ``__dlpack__`` attribute
            The object on which to invoke the DLPack protocol
        stream : int, optional
            The stream on which the tensor represented by obj should be made
            safe for use. This value is passed to the object's ``__dlpack__``
            method; however, the CBuffer does not keep any record of this (i.e.,
            the caller is responsible for creating a sync event after creating one
            or more buffers in this way).
        """
    @staticmethod
    def from_pybuffer(obj) -> CBuffer:
        """Create a CBuffer using the Python buffer protocol

        Wraps a buffer using the Python buffer protocol as a CBuffer that can be
        used to create an array.

        Parameters
        ----------
        obj : buffer-like
            The object on which to invoke the Python buffer protocol
        """
    def view(self):
        """Export this buffer as a CBufferView

        Returns a :class:`CBufferView` of this buffer. After calling this
        method, the original CBuffer will be invalidated and cannot be used.
        In general, the view of the buffer should be used to consume a buffer
        (whereas the CBuffer is primarily used to wrap an existing object in
        a way that it can be used to build a :class:`CArray`).
        """
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

class CBufferBuilder:
    """Wrapper around writable CPU buffer content

    This class provides a growable type-aware buffer that can be used
    to create a typed buffer from a Python iterable. This method of
    creating a buffer is usually slower than constructors like
    ``array.array()`` or ``numpy.array()``; however, this class supports
    all Arrow types with a single data buffer (e.g., boolean bitmaps,
    float16, intervals, fixed-size binary), some of which are not supported
    by other constructors.
    """

    capacity_bytes: Incomplete
    format: Incomplete
    itemsize: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def advance(self, additional_bytes):
        """Manually increase :attr:`size_bytes` by ``additional_bytes``

        This can be used after writing to the buffer using the buffer protocol
        to ensure that :attr:`size_bytes` accurately reflects the number of
        bytes written to the buffer.
        """
    def finish(self):
        """Finish building this buffer

        Performs any steps required to finish building this buffer and
        returns the result. Any behaviour resulting from calling methods
        on this object after it has been finished is not currently
        defined (but should not crash).
        """
    def reserve_bytes(self, additional_bytes):
        """Ensure that the underlying buffer has space for ``additional_bytes``
        more bytes to be written"""
    def set_data_type(self, type_id, element_size_bits=...):
        """Set the data type used to interpret elements in :meth:`write_elements`."""
    def set_format(self, format):
        """Set the Python buffer format used to interpret elements in
        :meth:`write_elements`.
        """
    def write(self, content):
        """Write bytes to this buffer

        Writes the bytes of ``content`` without considering the element type of
        ``content`` or the element type of this buffer.

        This method returns the number of bytes that were written.
        """
    def write_elements(self, obj):
        """ "Write an iterable of elements to this buffer

        Writes the elements of iterable ``obj`` according to the binary
        representation specified by :attr:`format`. This is currently
        powered by ``struct.pack_into()`` except when building bitmaps
        where an internal implementation is used.

        This method returns the number of elements that were written.
        """
    def write_fill(self, value, size_bytes):
        """Write fill bytes to this buffer

        Appends the byte ``value`` to this buffer ``size_bytes`` times.
        """
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

class CBufferView:
    """Wrapper for Array buffer content

    This object is a Python wrapper around a buffer held by an Array.
    It implements the Python buffer protocol and is best accessed through
    another implementor (e.g., `np.array(array_view.buffers[1])`)). Note that
    this buffer content does not apply any parent offset.
    """

    data_type: Incomplete
    data_type_id: Incomplete
    device: Incomplete
    element_size_bits: Incomplete
    format: Incomplete
    itemsize: Incomplete
    n_elements: Incomplete
    size_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def copy(self, offset=..., length=...): ...
    def copy_into(self, dest, offset=..., length=..., dest_offset=...): ...
    def element(self, i): ...
    def elements(self, offset=..., length=...): ...
    def unpack_bits(self, offset=..., length=...): ...
    def unpack_bits_into(self, dest, offset=..., length=..., dest_offset=...): ...
    def __buffer__(self, *args, **kwargs):
        """Return a buffer object that exposes the underlying memory of the object."""
    def __dlpack__(self, stream=...):
        """
        Export CBufferView as a DLPack capsule.

        Parameters
        ----------
        stream : int, optional
            A Python integer representing a pointer to a stream.
            Stream is provided by the consumer to the producer to instruct the producer
            to ensure that operations can safely be performed on the array.

        Returns
        -------
        capsule : PyCapsule
            A DLPack capsule for the array, pointing to a DLManagedTensor.
        """
    def __dlpack_device__(self):
        """
        Return the DLPack device tuple this CBufferView resides on.

        Returns
        -------
        tuple : Tuple[int, int]
            Tuple with index specifying the type of the device (where
            CPU = 1, see python/src/nanoarrow/dpack_abi.h) and index of the
            device which is 0 by default for CPU.
        """
    def __getitem__(self, index):
        """Return self[key]."""
    def __iter__(self):
        """Implement iter(self)."""
    def __len__(self) -> int:
        """Return len(self)."""
    def __reduce__(self): ...
    def __release_buffer__(self, *args, **kwargs):
        """Release the buffer object that exposes the underlying memory of the object."""

class NoneAwareWrapperIterator:
    """Nullable iterator wrapper

    This class wraps an iterable ``obj`` that might contain ``None`` values
    such that the iterable provided by this class contains "empty" (but valid)
    values. After ``obj`` has been completely consumed, one can call
    ``finish()`` to obtain the resulting bitmap. This is useful for passing
    iterables that might contain None to tools that cannot handle them
    (e.g., struct.pack(), array.array()).
    """

    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def finish(self):
        """Obtain the total count, null count, and validity bitmap after
        consuming this iterable."""
    def reserve(self, additional_elements): ...
    def __iter__(self):
        """Implement iter(self)."""
    def __reduce__(self): ...
