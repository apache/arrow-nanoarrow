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

# cython: language_level = 3

from libc.stdint cimport uintptr_t, int8_t, uint8_t, uint16_t, int64_t
from libc.string cimport memcpy
from libc.stdio cimport snprintf
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_IsValid
from cpython cimport (
    Py_buffer,
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBuffer_ToContiguous,
    PyBuffer_FillInfo,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_FORMAT,
    PyBUF_WRITABLE,
    PyErr_WriteUnraisable,
)
from cpython.ref cimport Py_INCREF, Py_DECREF

from nanoarrow_c cimport (
    NANOARROW_OK,
    ArrowMalloc,
    ArrowFree,
    ArrowType,
    ArrowTypeString,
    ArrowBitGet,
    ArrowBitsUnpackInt8,
    ArrowBufferReserve,
    ArrowBufferAppendFill,
    ArrowBufferAppendInt8,
    ArrowBitmapInit,
    ArrowBitmapReset,
    ArrowBitmap,
    ArrowBitmapReserve,
    ArrowBitmapAppend,
    ArrowBitmapAppendUnsafe,
    ArrowBuffer,
    ArrowBufferMove,
)

from nanoarrow_device_c cimport (
    ARROW_DEVICE_CPU,
    ARROW_DEVICE_CUDA,
    ArrowDevice,
)

from nanoarrow_dlpack cimport (
    DLDataType,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    DLTensor,
    kDLCPU,
    kDLFloat,
    kDLInt,
    kDLUInt
)

from nanoarrow cimport _utils
from nanoarrow cimport _types
from nanoarrow._device cimport CSharedSyncEvent, Device

from struct import unpack_from, iter_unpack, calcsize, Struct

from nanoarrow import _repr_utils
from nanoarrow._device import DEVICE_CPU


cdef void pycapsule_dlpack_deleter(object dltensor) noexcept:
    cdef DLManagedTensor* dlm_tensor

    # Do nothing if the capsule has been consumed
    if PyCapsule_IsValid(dltensor, "used_dltensor"):
        return

    dlm_tensor = <DLManagedTensor*>PyCapsule_GetPointer(dltensor, 'dltensor')
    if dlm_tensor == NULL:
        PyErr_WriteUnraisable(dltensor)
    # The deleter can be NULL if there is no way for the caller
    # to provide a reasonable destructor
    elif dlm_tensor.deleter:
        dlm_tensor.deleter(dlm_tensor)


cdef void view_dlpack_deleter(DLManagedTensor* tensor) noexcept with gil:
    if tensor.manager_ctx is NULL:
        return
    Py_DECREF(<CBufferView>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    ArrowFree(tensor)


cdef DLDataType view_to_dlpack_data_type(CBufferView view):
    cdef DLDataType dtype
    # Define DLDataType struct
    if _types.is_unsigned_integer(view.data_type_id):
        dtype.code = kDLUInt
    elif _types.is_signed_integer(view.data_type_id):
        dtype.code = kDLInt
    elif _types.is_floating_point(view.data_type_id):
        dtype.code = kDLFloat
    elif _types.equal(view.data_type_id, _types.BOOL):
        raise ValueError('Bit-packed boolean data type not supported by DLPack.')
    else:
        raise ValueError('DataType is not compatible with DLPack spec: ' + view.data_type)
    dtype.lanes = <uint16_t>1
    dtype.bits = <uint8_t>(view._element_size_bits)

    return dtype

cdef int dlpack_data_type_to_arrow(DLDataType dtype):
    if dtype.code == kDLInt:
        if dtype.bits == 8:
            return _types.INT8
        elif dtype.bits == 16:
            return _types.INT16
        elif dtype.bits == 32:
            return _types.INT32
        elif dtype.bits == 64:
            return _types.INT64
    elif dtype.code == kDLUInt:
        if dtype.bits == 8:
            return _types.UINT8
        elif dtype.bits == 16:
            return _types.UINT16
        elif dtype.bits == 32:
            return _types.UINT32
        elif dtype.bits == 64:
            return _types.UINT64
    elif dtype.code == kDLFloat:
        if dtype.bits == 16:
            return _types.HALF_FLOAT
        elif dtype.bits == 32:
            return _types.FLOAT
        elif dtype.bits == 64:
            return _types.DOUBLE

    raise ValueError("Can't convert dlpack data type to Arrow type")

cdef object view_to_dlpack(CBufferView view, stream=None):
    # Define DLDevice and DLDataType struct and
    # with that check for data type support first
    cdef DLDevice device = view_to_dlpack_device(view)
    cdef DLDataType dtype = view_to_dlpack_data_type(view)

    # Allocate memory for DLManagedTensor
    cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>ArrowMalloc(sizeof(DLManagedTensor))
    # Define DLManagedTensor struct
    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*>view._ptr.data.data
    dl_tensor.ndim = 1

    dl_tensor.shape = &view._n_elements
    dl_tensor.strides = NULL
    dl_tensor.byte_offset = 0

    dl_tensor.device = device
    dl_tensor.dtype = dtype

    dlm_tensor.manager_ctx = <void*>view
    Py_INCREF(view)
    dlm_tensor.deleter = view_dlpack_deleter

    # stream has a DLPack + device specific interpretation

    # nanoarrow_device needs a CUstream* (i.e., a CUstream_st**), but dlpack
    # gives us a CUstream_st*.
    cdef void* cuda_pstream

    if view._event.device is DEVICE_CPU:
        if stream is not None and stream != -1:
            raise ValueError("dlpack stream must be None or -1 for the CPU device")
    elif view._event.device.device_type_id == ARROW_DEVICE_CUDA:
        if stream == 0:
            raise ValueError("dlpack stream value of 0 is not permitted for CUDA")
        elif stream == -1:
            # Sentinel for "do not synchronize"
            pass
        elif stream in (1, 2):
            # Technically we are mixing the per-thread and legacy default streams here;
            # however, the nanoarrow_device API currently has no mechanism to expose
            # a pointer to these streams specifically.
            cuda_pstream = <void*>0
            view._event.synchronize_stream(<uintptr_t>&cuda_pstream)
        else:
            # Otherwise, this is a CUstream** (i.e., CUstream_st*)
            cuda_pstream = <void*><uintptr_t>stream
            view._event.synchronize_stream(<uintptr_t>&cuda_pstream)

    return PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_dlpack_deleter)


cdef DLDevice view_to_dlpack_device(CBufferView view):
    cdef DLDevice device

    # Check data type support
    if _types.equal(view.data_type_id, _types.BOOL):
        raise ValueError('Bit-packed boolean data type not supported by DLPack.')
    elif (
        not _types.is_unsigned_integer(view.data_type_id)
        and not _types.is_signed_integer(view.data_type_id)
        and not _types.is_floating_point(view.data_type_id)
    ):
        raise ValueError('DataType is not compatible with DLPack spec: ' + view.data_type)

    # Define DLDevice struct
    cdef ArrowDevice* arrow_device = view._event.device._ptr
    if arrow_device.device_type is ARROW_DEVICE_CPU:
        # DLPack uses 0 for the CPU device id where Arrow uses -1
        device.device_type = kDLCPU
        device.device_id =  0
    else:
        # Otherwise, Arrow's device identifiers and types are intentionally
        # identical to DLPack
        device.device_type = <DLDeviceType>arrow_device.device_type
        device.device_id = arrow_device.device_id

    return device


cdef bint dlpack_strides_are_contiguous(DLTensor* dl_tensor):
    if dl_tensor.strides == NULL:
        return True

    if dl_tensor.ndim != 1:
        raise NotImplementedError("Contiguous stride check not implemented for ndim != 1")

    # DLTensor strides are in elemements, not bytes
    return dl_tensor.strides[0] == 1


cdef class CBufferView:
    """Wrapper for Array buffer content

    This object is a Python wrapper around a buffer held by an Array.
    It implements the Python buffer protocol and is best accessed through
    another implementor (e.g., `np.array(array_view.buffers[1])`)). Note that
    this buffer content does not apply any parent offset.
    """

    def __cinit__(self, object base, uintptr_t addr, int64_t size_bytes,
                  ArrowType data_type,
                  Py_ssize_t element_size_bits, CSharedSyncEvent event):
        self._base = base
        self._ptr.data.data = <void*>addr
        self._ptr.size_bytes = size_bytes
        self._data_type = data_type
        self._event = event
        self._format[0] = 0
        self._element_size_bits = _types.to_format(
            self._data_type,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._strides = self._item_size()
        self._shape = self._ptr.size_bytes // self._strides
        if _types.equal(self._data_type, _types.BOOL):
            self._n_elements = self._shape * 8
        else:
            self._n_elements = self._shape

    def _addr(self):
        return <uintptr_t>self._ptr.data.data

    @property
    def device(self):
        return self._event.device

    @property
    def element_size_bits(self):
        return self._element_size_bits

    @property
    def size_bytes(self):
        return self._ptr.size_bytes

    @property
    def data_type_id(self):
        return self._data_type

    @property
    def data_type(self):
        return ArrowTypeString(self._data_type).decode("UTF-8")

    @property
    def format(self):
        return self._format.decode("UTF-8")

    @property
    def itemsize(self):
        return self._strides

    def __len__(self):
        return self._shape

    def __getitem__(self, int64_t i):
        if i < 0 or i >= self._shape:
            raise IndexError(f"Index {i} out of range")
        cdef int64_t offset = self._strides * i
        value = unpack_from(self.format, buffer=self, offset=offset)
        if len(value) == 1:
            return value[0]
        else:
            return value

    def __iter__(self):
        return self._iter_dispatch(0, len(self))

    def _iter_dispatch(self, int64_t offset, int64_t length):
        if offset < 0 or length < 0 or (offset + length) > len(self):
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with length {len(self)}"
            )
        # memoryview's implementation is very fast but not always possible (half float, fixed-size binary, interval)
        if _types.one_of(
            self._data_type,
            (
                _types.HALF_FLOAT,
                _types.INTERVAL_DAY_TIME,
                _types.INTERVAL_MONTH_DAY_NANO,
                _types.DECIMAL128,
                _types.DECIMAL256
            )
        ) or (
            _types.equal(self._data_type, _types.BINARY) and self._element_size_bits != 0
        ):
            return self._iter_struct(offset, length)
        else:
            return self._iter_memoryview(offset, length)

    def _iter_memoryview(self, int64_t offset, int64_t length):
        return iter(memoryview(self)[offset:(offset + length)])

    def _iter_struct(self, int64_t offset, int64_t length):
        for value in iter_unpack(self.format, self):
            if len(value) == 1:
                yield value[0]
            else:
                yield value

    @property
    def n_elements(self):
        return self._n_elements

    def element(self, i):
        if _types.equal(self._data_type, _types.BOOL):
            if i < 0 or i >= self.n_elements:
                raise IndexError(f"Index {i} out of range")
            return ArrowBitGet(self._ptr.data.as_uint8, i)
        else:
            return self[i]

    def elements(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        if offset < 0 or length < 0 or (offset + length) > self.n_elements:
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with {self.n_elements} elements"
            )

        if _types.equal(self._data_type, _types.BOOL):
            return self._iter_bitmap(offset, length)
        else:
            return self._iter_dispatch(offset, length)

    def copy_into(self, dest, offset=0, length=None, dest_offset=0):
        if length is None:
            length = self.n_elements

        cdef Py_buffer buffer
        PyObject_GetBuffer(dest, &buffer, PyBUF_WRITABLE | PyBUF_ANY_CONTIGUOUS)

        cdef int64_t c_offset = offset
        cdef int64_t c_length = length
        cdef int64_t c_item_size = self.itemsize
        cdef int64_t c_dest_offset = dest_offset
        self._check_copy_into_bounds(&buffer, c_offset, c_length, dest_offset, c_item_size)

        cdef uint8_t* dest_uint8 = <uint8_t*>buffer.buf
        cdef int64_t dest_offset_bytes = c_dest_offset * c_item_size
        cdef int64_t src_offset_bytes = c_offset * c_item_size
        cdef int64_t bytes_to_copy = c_length * c_item_size

        memcpy(
            &(dest_uint8[dest_offset_bytes]),
            &(self._ptr.data.as_uint8[src_offset_bytes]),
            bytes_to_copy
        )

        PyBuffer_Release(&buffer)
        return bytes_to_copy

    def unpack_bits_into(self, dest, offset=0, length=None, dest_offset=0):
        if not _types.equal(self._data_type, _types.BOOL):
            raise ValueError("Can't unpack non-boolean buffer")

        if length is None:
            length = self.n_elements

        cdef Py_buffer buffer
        PyObject_GetBuffer(dest, &buffer, PyBUF_WRITABLE | PyBUF_ANY_CONTIGUOUS)
        self._check_copy_into_bounds(&buffer, offset, length, dest_offset, 1)

        ArrowBitsUnpackInt8(
            self._ptr.data.as_uint8,
            offset,
            length,
            &(<int8_t*>buffer.buf)[dest_offset]
        )

        PyBuffer_Release(&buffer)
        return length

    def unpack_bits(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        out = CBufferBuilder().set_format("?")
        out.reserve_bytes(length)
        self.unpack_bits_into(out, offset, length)
        out.advance(length)
        return out.finish()

    def copy(self, offset=0, length=None):
        if length is None:
            length = self.n_elements

        cdef int64_t bytes_to_copy = length * self.itemsize
        out = CBufferBuilder().set_data_type(self.data_type_id)
        out.reserve_bytes(bytes_to_copy)
        self.copy_into(out, offset, length)
        out.advance(bytes_to_copy)
        return out.finish()

    cdef _check_copy_into_bounds(self, Py_buffer* dest, int64_t offset, int64_t length,
                                 int64_t dest_offset, int64_t dest_itemsize):
        if offset < 0 or length < 0 or (offset + length) > self.n_elements:
            PyBuffer_Release(dest)
            raise IndexError(
                f"offset {offset} and length {length} do not describe a valid slice "
                f"of buffer with {self.n_elements} elements"
            )

        if dest.itemsize != 1 and dest.itemsize != dest_itemsize:
            raise ValueError(
                "Destination buffer must have itemsize == 1 or "
                f"itemsize == {dest_itemsize}"
            )

        cdef int64_t dest_offset_bytes = dest_offset * dest_itemsize
        cdef int64_t bytes_to_copy = dest_itemsize * length
        if dest_offset < 0 or dest.len < (dest_offset_bytes + bytes_to_copy):
            buffer_len = dest.len
            PyBuffer_Release(dest)
            raise IndexError(
                f"Can't unpack {length} elements into buffer of size {buffer_len} "
                f"with dest_offset = {dest_offset}"
            )

    def _iter_bitmap(self, int64_t offset, int64_t length):
        cdef uint8_t item
        cdef int64_t i

        if offset % 8 == 0:
            first_byte = offset // 8
            last_byte = self._shape
            i = 0

            for byte_i in range(first_byte, last_byte):
                item = self._ptr.data.as_uint8[byte_i]
                for j in range(8):
                    yield (item & (<uint8_t>1 << j)) != 0
                    i += 1
                    if i >= length:
                        return
        else:
            for i in range(length):
                yield ArrowBitGet(self._ptr.data.as_uint8, offset + i) != 0


    cdef Py_ssize_t _item_size(self):
        if self._element_size_bits < 8:
            return 1
        else:
            return self._element_size_bits // 8


    def __dlpack__(self, stream=None):
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
        # Note: parent offset not applied!
        return view_to_dlpack(self, stream)


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
        cdef DLDevice dlpack_device = view_to_dlpack_device(self)
        return dlpack_device.device_type, dlpack_device.device_id

    # These are special methods, which can't be cdef and we can't
    # call them from elsewhere. We implement the logic for the buffer
    # protocol separately so we can re-use it in the CBuffer.
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        self._do_getbuffer(buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        self._do_releasebuffer(buffer)

    cdef _do_getbuffer(self, Py_buffer *buffer, int flags):
        if self.device is not DEVICE_CPU:
            raise RuntimeError("CBufferView is not a CPU buffer")

        if flags & PyBUF_WRITABLE:
            raise BufferError("CBufferView does not support PyBUF_WRITABLE")

        buffer.buf = <void*>self._ptr.data.data

        if flags & PyBUF_FORMAT:
            buffer.format = self._format
        else:
            buffer.format = NULL

        buffer.internal = NULL
        buffer.itemsize = self._strides
        buffer.len = self._ptr.size_bytes
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = &self._shape
        buffer.strides = &self._strides
        buffer.suboffsets = NULL

    cdef _do_releasebuffer(self, Py_buffer* buffer):
        pass

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_buffer")
        return f"{class_label}({_repr_utils.buffer_view_repr(self)})"


cdef class CBuffer:
    """Wrapper around readable owned buffer content

    Like the CBufferView, the CBuffer represents readable buffer content; however,
    unlike the CBufferView, the CBuffer always represents a valid ArrowBuffer C object.
    Whereas the CBufferView is primarily concerned with accessing the contents of a
    buffer, the CBuffer is primarily concerned with managing ownership of an external
    buffer such that it exported as an Arrow array.
    """

    def __cinit__(self):
        self._base = None
        self._ptr = NULL
        self._data_type = <ArrowType>(_types.BINARY)
        self._element_size_bits = 0
        self._device = DEVICE_CPU
        # Set initial format to "B" (Cython makes this hard)
        self._format[0] = 66
        self._format[1] = 0
        self._get_buffer_count = 0
        self._view = CBufferView(
            None, 0,
            0, _types.BINARY, 0,
            CSharedSyncEvent(self._device)
        )

    cdef _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("CBuffer is not valid")

    cdef _assert_buffer_count_zero(self):
        if self._get_buffer_count != 0:
            raise RuntimeError(
                f"CBuffer already open ({self._get_buffer_count} ",
                f"references, {self._writable_get_buffer_count} writable)")

    cdef _populate_view(self):
        self._assert_valid()
        self._assert_buffer_count_zero()
        self._view = CBufferView(
            self._base, <uintptr_t>self._ptr.data,
            self._ptr.size_bytes, self._data_type, self._element_size_bits,
            CSharedSyncEvent(self._device)
        )

        snprintf(self._view._format, sizeof(self._view._format), "%s", self._format)

    def view(self):
        """Export this buffer as a CBufferView

        Returns a :class:`CBufferView` of this buffer. After calling this
        method, the original CBuffer will be invalidated and cannot be used.
        In general, the view of the buffer should be used to consume a buffer
        (whereas the CBuffer is primarily used to wrap an existing object in
        a way that it can be used to build a :class:`CArray`).
        """
        self._assert_valid()
        self._assert_buffer_count_zero()
        cdef ArrowBuffer* new_ptr
        self._view._base = _utils.alloc_c_buffer(&new_ptr)
        ArrowBufferMove(self._ptr, new_ptr)
        self._ptr = NULL
        return self._view

    @staticmethod
    def empty():
        """Create an empty CBuffer"""
        cdef CBuffer out = CBuffer()
        out._base = _utils.alloc_c_buffer(&out._ptr)
        return out

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
        cdef CBuffer out = CBuffer()
        out._base = _utils.alloc_c_buffer(&out._ptr)
        out._set_format(_utils.c_buffer_set_pybuffer(obj, &out._ptr))
        out._device = DEVICE_CPU
        out._populate_view()
        return out

    @staticmethod
    def from_dlpack(obj, stream=None) -> CBuffer:
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
        capsule = obj.__dlpack__(stream=stream)
        cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>PyCapsule_GetPointer(
            capsule, "dltensor"
        )
        cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor

        if not dlpack_strides_are_contiguous(dl_tensor):
            raise ValueError("Non-contiguous dlpack strides not supported")

        cdef Device device = Device.resolve(
            dl_tensor.device.device_type,
            dl_tensor.device.device_id
        )
        cdef int arrow_type = dlpack_data_type_to_arrow(dl_tensor.dtype)
        cdef uint8_t* data_ptr = <uint8_t*>dl_tensor.data + dl_tensor.byte_offset

        cdef int64_t size_bytes = 1
        cdef int64_t element_size_bytes = dl_tensor.dtype.bits // 8
        for i in range(dl_tensor.ndim):
            size_bytes *= dl_tensor.shape[i] * element_size_bytes

        cdef CBuffer out = CBuffer()

        out._base = _utils.alloc_c_buffer(&out._ptr)
        _utils.c_buffer_set_pyobject(capsule, data_ptr, size_bytes, &out._ptr)
        out._set_data_type(arrow_type)
        out._device = device
        out._populate_view()

        return out

    def _set_format(self, str format):
        self._assert_buffer_count_zero()

        element_size_bytes, data_type = _types.from_format(format)
        self._data_type = data_type
        self._element_size_bits = element_size_bytes * 8
        format_bytes = format.encode("UTF-8")
        snprintf(self._format, sizeof(self._format), "%s", <const char*>format_bytes)
        self._populate_view()
        return self

    def _set_data_type(self, ArrowType type_id, int element_size_bits=0):
        self._assert_buffer_count_zero()

        self._element_size_bits = _types.to_format(
            type_id,
            element_size_bits,
            sizeof(self._format),
            self._format
        )
        self._data_type = type_id

        self._populate_view()
        return self

    def _addr(self):
        self._assert_valid()
        return <uintptr_t>self._ptr.data

    @property
    def _get_buffer_count(self):
        return self._get_buffer_count

    @property
    def size_bytes(self):
        self._assert_valid()
        return self._ptr.size_bytes

    @property
    def data_type(self):
        self._assert_valid()
        return ArrowTypeString(self._data_type).decode("UTF-8")

    @property
    def data_type_id(self):
        self._assert_valid()
        return self._data_type

    @property
    def element_size_bits(self):
        self._assert_valid()
        return self._element_size_bits

    @property
    def itemsize(self):
        self._assert_valid()
        return self._view.itemsize

    @property
    def format(self):
        self._assert_valid()
        return self._format.decode("UTF-8")

    @property
    def device(self):
        self._assert_valid()
        return self._view.device

    def __len__(self):
        self._assert_valid()
        return len(self._view)

    def __getitem__(self, k):
        self._assert_valid()
        return self._view[k]

    def __iter__(self):
        self._assert_valid()
        return iter(self._view)

    @property
    def n_elements(self):
        self._assert_valid()
        return self._view.n_elements

    def element(self, i):
        self._assert_valid()
        return self._view.element(i)

    def elements(self, offset=0, length=None):
        self._assert_valid()
        return self._view.elements(offset, length)

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self._assert_valid()
        self._view._do_getbuffer(buffer, flags)
        self._get_buffer_count += 1

    def __releasebuffer__(self, Py_buffer* buffer):
        if self._get_buffer_count <= 0:
            raise RuntimeError("CBuffer buffer reference count underflow (releasebuffer)")

        self._view._do_releasebuffer(buffer)
        self._get_buffer_count -= 1

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_buffer")
        if self._ptr == NULL:
            return f"{class_label}(<invalid>)"

        return f"{class_label}({_repr_utils.buffer_view_repr(self._view)})"


cdef class CBufferBuilder:
    """Wrapper around writable CPU buffer content

    This class provides a growable type-aware buffer that can be used
    to create a typed buffer from a Python iterable. This method of
    creating a buffer is usually slower than constructors like
    ``array.array()`` or ``numpy.array()``; however, this class supports
    all Arrow types with a single data buffer (e.g., boolean bitmaps,
    float16, intervals, fixed-size binary), some of which are not supported
    by other constructors.
    """
    cdef CBuffer _buffer
    cdef bint _locked

    def __cinit__(self):
        self._buffer = CBuffer.empty()
        self._locked = False

    cdef _assert_unlocked(self):
        if self._locked:
            raise BufferError("CBufferBuilder is locked")

    # Implement the buffer protocol so that this object can be used as
    # the argument to Struct.readinto() (or perhaps written to by
    # an independent library).
    def __getbuffer__(self, Py_buffer* buffer, int flags):
        self._assert_unlocked()
        PyBuffer_FillInfo(
            buffer,
            self,
            self._buffer._ptr.data,
            self._buffer._ptr.capacity_bytes,
            0,
            flags
        )
        self._locked = True

    def __releasebuffer__(self, Py_buffer* buffer):
        self._locked = False

    def set_data_type(self, ArrowType type_id, int element_size_bits=0):
        """Set the data type used to interpret elements in :meth:`write_elements`."""
        self._buffer._set_data_type(type_id, element_size_bits)
        return self

    def set_format(self, str format):
        """Set the Python buffer format used to interpret elements in
        :meth:`write_elements`.
        """
        self._buffer._set_format(format)
        return self

    @property
    def format(self):
        """The ``struct`` format code of the underlying buffer"""
        return self._buffer._format.decode()

    @property
    def size_bytes(self):
        """The number of bytes that have been written to this buffer"""
        return self._buffer.size_bytes

    @property
    def itemsize(self):
        return self._buffer.itemsize

    def __len__(self):
        return self._buffer.size_bytes // self.itemsize

    @property
    def capacity_bytes(self):
        """The number of bytes allocated in the underlying buffer"""
        return self._buffer._ptr.capacity_bytes

    def reserve_bytes(self, int64_t additional_bytes):
        """Ensure that the underlying buffer has space for ``additional_bytes``
        more bytes to be written"""
        self._assert_unlocked()
        cdef int code = ArrowBufferReserve(self._buffer._ptr, additional_bytes)
        _utils.Error.raise_error_not_ok("ArrowBufferReserve()", code)
        return self

    def advance(self, int64_t additional_bytes):
        """Manually increase :attr:`size_bytes` by ``additional_bytes``

        This can be used after writing to the buffer using the buffer protocol
        to ensure that :attr:`size_bytes` accurately reflects the number of
        bytes written to the buffer.
        """
        cdef int64_t new_size = self._buffer._ptr.size_bytes + additional_bytes
        if new_size < 0 or new_size > self._buffer._ptr.capacity_bytes:
            raise IndexError(f"Can't advance {additional_bytes} from {self.size_bytes}")

        self._buffer._ptr.size_bytes = new_size
        return self

    def write_fill(self, uint8_t value, int64_t size_bytes):
        """Write fill bytes to this buffer

        Appends the byte ``value`` to this buffer ``size_bytes`` times.
        """
        self._assert_unlocked()
        cdef int code = ArrowBufferAppendFill(self._buffer._ptr, value, size_bytes)
        _utils.Error.raise_error_not_ok("ArrowBufferAppendFill", code)

    def write(self, content):
        """Write bytes to this buffer

        Writes the bytes of ``content`` without considering the element type of
        ``content`` or the element type of this buffer.

        This method returns the number of bytes that were written.
        """
        self._assert_unlocked()

        cdef Py_buffer buffer
        cdef int64_t out
        PyObject_GetBuffer(content, &buffer, PyBUF_ANY_CONTIGUOUS)

        # TODO: Check for single dimension?

        cdef int code = ArrowBufferReserve(self._buffer._ptr, buffer.len)
        if code != NANOARROW_OK:
            PyBuffer_Release(&buffer)
            _utils.Error.raise_error("ArrowBufferReserve()", code)

        code = PyBuffer_ToContiguous(
            self._buffer._ptr.data + self._buffer._ptr.size_bytes,
            &buffer,
            buffer.len,
            # 'C' (not sure how to pass a character literal here)
            43
        )
        out = buffer.len
        PyBuffer_Release(&buffer)
        _utils.Error.raise_error_not_ok("PyBuffer_ToContiguous()", code)

        self._buffer._ptr.size_bytes += out
        return out

    def write_elements(self, obj):
        """"Write an iterable of elements to this buffer

        Writes the elements of iterable ``obj`` according to the binary
        representation specified by :attr:`format`. This is currently
        powered by ``struct.pack_into()`` except when building bitmaps
        where an internal implementation is used.

        This method returns the number of elements that were written.
        """
        self._assert_unlocked()

        # Boolean arrays need their own writer since Python provides
        # no helpers for bitpacking
        if _types.equal(self._buffer._data_type, _types.BOOL):
            return self._write_bits(obj)

        cdef int64_t n_values = 0
        cdef int64_t bytes_per_element = calcsize(self._buffer.format)
        cdef int code

        struct_obj = Struct(self._buffer._format)
        pack = struct_obj.pack
        write = self.write

        # If an object has a length, we can avoid extra allocations
        if hasattr(obj, "__len__"):
            code = ArrowBufferReserve(self._buffer._ptr, bytes_per_element * len(obj))
            _utils.Error.raise_error_not_ok("ArrowBufferReserve()", code)

        # Types whose Python representation is a tuple need a slightly different
        # pack_into() call. if code != NANOARROW_OK is used instead of
        # Error.raise_error_not_ok() Cython can avoid the extra function call
        # and this is a very tight loop.
        if _types.one_of(
            self._buffer._data_type,
            (_types.INTERVAL_DAY_TIME, _types.INTERVAL_MONTH_DAY_NANO)
        ):
            for item in obj:
                write(pack(*item))
                n_values += 1

        else:
            for item in obj:
                write(pack(item))
                n_values += 1

        return n_values

    cdef _write_bits(self, obj):
        if self._buffer._ptr.size_bytes != 0:
            raise NotImplementedError("Append to bitmap that has already been appended to")

        cdef char buffer_item = 0
        cdef int buffer_item_i = 0
        cdef int code
        cdef int64_t n_values = 0
        for item in obj:
            n_values += 1
            if item:
                buffer_item |= (<char>1 << buffer_item_i)

            buffer_item_i += 1
            if buffer_item_i == 8:
                code = ArrowBufferAppendInt8(self._buffer._ptr, buffer_item)
                _utils.Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)
                buffer_item = 0
                buffer_item_i = 0

        if buffer_item_i != 0:
            code = ArrowBufferAppendInt8(self._buffer._ptr, buffer_item)
            _utils.Error.raise_error_not_ok("ArrowBufferAppendInt8()", code)

        return n_values

    def finish(self):
        """Finish building this buffer

        Performs any steps required to finish building this buffer and
        returns the result. Any behaviour resulting from calling methods
        on this object after it has been finished is not currently
        defined (but should not crash).
        """
        self._assert_unlocked()

        cdef CBuffer out = self._buffer
        out._populate_view()

        self._buffer = CBuffer.empty()
        return out

    def __repr__(self):
        class_label = _repr_utils.make_class_label(self, module="nanoarrow.c_buffer")
        return f"{class_label}({self.size_bytes}/{self.capacity_bytes})"


cdef class NoneAwareWrapperIterator:
    """Nullable iterator wrapper

    This class wraps an iterable ``obj`` that might contain ``None`` values
    such that the iterable provided by this class contains "empty" (but valid)
    values. After ``obj`` has been completely consumed, one can call
    ``finish()`` to obtain the resulting bitmap. This is useful for passing
    iterables that might contain None to tools that cannot handle them
    (e.g., struct.pack(), array.array()).
    """
    cdef ArrowBitmap _bitmap
    cdef object _obj
    cdef object _value_if_none
    cdef int64_t _valid_count
    cdef int64_t _item_count

    def __cinit__(self, obj, type_id, item_size_bytes=0):
        ArrowBitmapInit(&self._bitmap)
        self._obj = iter(obj)

        self._value_if_none = self._get_value_if_none(type_id, item_size_bytes)
        self._valid_count = 0
        self._item_count = 0

    def __dealloc__(self):
        ArrowBitmapReset(&self._bitmap)

    def reserve(self, int64_t additional_elements):
        cdef int code = ArrowBitmapReserve(&self._bitmap, additional_elements)
        _utils.Error.raise_error_not_ok(self, code)

    def _get_value_if_none(self, type_id, item_size_bytes=0):
        if _types.equal(type_id, _types.INTERVAL_DAY_TIME):
            return (0, 0)
        elif _types.equal(type_id, _types.INTERVAL_MONTH_DAY_NANO):
            return (0, 0, 0)
        elif _types.equal(type_id, _types.BOOL):
            return False
        elif _types.one_of(type_id, (_types.BINARY, _types.FIXED_SIZE_BINARY)):
            return b"\x00" * item_size_bytes
        elif _types.one_of(type_id, (_types.HALF_FLOAT, _types.FLOAT, _types.DOUBLE)):
            return 0.0
        else:
            return 0

    cdef _append_to_validity(self, int is_valid):
        self._valid_count += is_valid
        self._item_count += 1

        # Avoid allocating a bitmap if all values seen so far are valid
        if self._valid_count == self._item_count:
            return

        # If the bitmap hasn't been allocated yet, allocate it now and
        # fill with 1s for all previous elements.
        cdef int code
        if self._bitmap.size_bits == 0 and self._item_count > 1:
            code = ArrowBitmapAppend(&self._bitmap, 1, self._item_count - 1)
            if code != NANOARROW_OK:
                _utils.Error.raise_error("ArrowBitmapAppend()", code)

        # Append this element to the bitmap
        code = ArrowBitmapAppend(&self._bitmap, is_valid, 1)
        if code != NANOARROW_OK:
            _utils.Error.raise_error("ArrowBitmapAppend()", code)

    def __iter__(self):
        for item in self._obj:
            if item is None:
                self._append_to_validity(0)
                yield self._value_if_none
            else:
                self._append_to_validity(1)
                yield item

    def finish(self):
        """Obtain the total count, null count, and validity bitmap after
        consuming this iterable."""
        cdef CBuffer validity
        null_count = self._item_count - self._valid_count

        # If we did allocate a bitmap, make sure the last few bits are zeroed
        if null_count > 0 and self._bitmap.size_bits % 8 != 0:
            ArrowBitmapAppendUnsafe(&self._bitmap, 0, self._bitmap.size_bits % 8)

        if null_count > 0:
            validity = CBuffer.empty()
            ArrowBufferMove(&self._bitmap.buffer, validity._ptr)
            validity._set_data_type(<ArrowType>_types.BOOL)

        return (
            self._item_count,
            null_count,
            validity if null_count > 0 else None
        )
