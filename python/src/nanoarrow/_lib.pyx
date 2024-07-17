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
# cython: linetrace=True

"""Low-level nanoarrow Python bindings

This Cython extension provides low-level Python wrappers around the
Arrow C Data and Arrow C Stream interface structs. In general, there
is one wrapper per C struct and pointer validity is managed by keeping
strong references to Python objects. These wrappers are intended to
be literal and stay close to the structure definitions: higher level
interfaces can and should be built in Python where it is faster to
iterate and where it is easier to create a better user experience
by default (i.e., classes, methods, and functions implemented in Python
generally have better autocomplete + documentation available to IDEs).
"""

from libc.stdint cimport uintptr_t, uint8_t, int64_t
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from cpython cimport (
    Py_buffer,
    PyObject_GetBuffer,
    PyBuffer_Release,
    PyBUF_ANY_CONTIGUOUS,
    PyBUF_FORMAT,
)
from nanoarrow_c cimport *
from nanoarrow_device_c cimport (
    ARROW_DEVICE_CPU,
    ArrowDeviceType,
    ArrowDeviceArray,
    ArrowDeviceArrayInit,
)

from nanoarrow._device cimport Device

from nanoarrow cimport _types
from nanoarrow._buffer cimport CBuffer, CBufferView
from nanoarrow._schema cimport CSchema, CLayout, assert_type_equal
from nanoarrow._utils cimport (
    alloc_c_array,
    alloc_c_array_stream,
    alloc_c_device_array,
    alloc_c_array_view,
    c_array_shallow_copy,
    c_device_array_shallow_copy,
    Error
)

from nanoarrow import _repr_utils
from nanoarrow._device import DEVICE_CPU, DeviceType


cdef class CArray:
    """Low-level ArrowArray wrapper

    This object is a literal wrapper around a read-only ArrowArray. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowArray structures are always released).

    See `nanoarrow.c_array()` for construction and usage examples.
    """
    cdef object _base
    cdef ArrowArray* _ptr
    cdef CSchema _schema
    cdef ArrowDeviceType _device_type
    cdef int _device_id

    @staticmethod
    def allocate(CSchema schema):
        cdef ArrowArray* c_array_out
        base = alloc_c_array(&c_array_out)
        return CArray(base, <uintptr_t>c_array_out, schema)

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowArray*>addr
        self._schema = schema
        self._device_type = ARROW_DEVICE_CPU
        self._device_id = 0

    cdef _set_device(self, ArrowDeviceType device_type, int64_t device_id):
        self._device_type = device_type
        self._device_id = device_id

    @staticmethod
    def _import_from_c_capsule(schema_capsule, array_capsule):
        """
        Import from a ArrowSchema and ArrowArray PyCapsule tuple.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        array_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_array' containing an
            ArrowArray pointer.
        """
        cdef:
            CSchema out_schema
            CArray out

        out_schema = CSchema._import_from_c_capsule(schema_capsule)
        out = CArray(
            array_capsule,
            <uintptr_t>PyCapsule_GetPointer(array_capsule, 'arrow_array'),
            out_schema
        )

        return out

    def __getitem__(self, k):
        self._assert_valid()

        if not isinstance(k, slice):
            raise TypeError(
                f"Can't subset CArray with object of type {type(k).__name__}")

        if k.step is not None:
            raise ValueError("Can't slice CArray with step")

        cdef int64_t start = 0 if k.start is None else k.start
        cdef int64_t stop = self._ptr.length if k.stop is None else k.stop
        if start < 0:
            start = self._ptr.length + start
        if stop < 0:
            stop = self._ptr.length + stop

        if start > self._ptr.length or stop > self._ptr.length or stop < start:
            raise IndexError(
                f"{k} does not describe a valid slice of CArray "
                f"with length {self._ptr.length}"
            )

        cdef ArrowArray* c_array_out
        base = alloc_c_array(&c_array_out)
        c_array_shallow_copy(self._base, self._ptr, c_array_out)

        c_array_out.offset = c_array_out.offset + start
        c_array_out.length = stop - start
        cdef CArray out = CArray(base, <uintptr_t>c_array_out, self._schema)
        out._set_device(self._device_type, self._device_id)
        return out

    def __arrow_c_array__(self, requested_schema=None):
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
        self._assert_valid()

        if self._device_type != ARROW_DEVICE_CPU:
            raise ValueError(
                "Can't invoke __arrow_c_array__ on non-CPU array "
                f"with device_type {self._device_type}")

        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        # Export a shallow copy pointing to the same data in a way
        # that ensures this object stays valid.

        # TODO optimize this to export a version where children are reference
        # counted and can be released separately
        cdef ArrowArray* c_array_out
        array_capsule = alloc_c_array(&c_array_out)
        c_array_shallow_copy(self._base, self._ptr, c_array_out)

        return self._schema.__arrow_c_schema__(), array_capsule

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("CArray is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("CArray is released")

    def view(self):
        device = Device.resolve(self._device_type, self._device_id)
        return CArrayView.from_array(self, device)

    @property
    def schema(self):
        return self._schema

    @property
    def device_type(self):
        return DeviceType(self._device_type)

    @property
    def device_type_id(self):
        return self._device_type

    @property
    def device_id(self):
        return self._device_id

    def __len__(self):
        self._assert_valid()
        return self._ptr.length

    @property
    def length(self):
        return len(self)

    @property
    def offset(self):
        self._assert_valid()
        return self._ptr.offset

    @property
    def null_count(self):
        self._assert_valid()
        return self._ptr.null_count

    @property
    def n_buffers(self):
        self._assert_valid()
        return self._ptr.n_buffers

    @property
    def buffers(self):
        self._assert_valid()
        return tuple(<uintptr_t>self._ptr.buffers[i] for i in range(self._ptr.n_buffers))

    @property
    def n_children(self):
        self._assert_valid()
        return self._ptr.n_children

    def child(self, int64_t i):
        self._assert_valid()
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")
        cdef CArray out = CArray(
            self._base,
            <uintptr_t>self._ptr.children[i],
            self._schema.child(i)
        )
        out._set_device(self._device_type, self._device_id)
        return out

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def dictionary(self):
        self._assert_valid()
        cdef CArray out
        if self._ptr.dictionary != NULL:
            out = CArray(self, <uintptr_t>self._ptr.dictionary, self._schema.dictionary)
            out._set_device(self._device_type, self._device_id)
            return out
        else:
            return None

    def __repr__(self):
        return _repr_utils.array_repr(self)


cdef class CArrayView:
    """Low-level ArrowArrayView wrapper

    This object is a literal wrapper around an ArrowArrayView. It provides field accessors
    that return Python objects and handles the structure lifecycle (i.e., initialized
    ArrowArrayView structures are always released).

    See `nanoarrow.c_array_view()` for construction and usage examples.
    """
    cdef object _base
    cdef object _array_base
    cdef ArrowArrayView* _ptr
    cdef Device _device

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayView*>addr
        self._device = DEVICE_CPU

    def _set_array(self, CArray array, Device device=DEVICE_CPU):
        cdef Error error = Error()
        cdef int code

        if device is DEVICE_CPU:
            code = ArrowArrayViewSetArray(self._ptr, array._ptr, &error.c_error)
        else:
            code = ArrowArrayViewSetArrayMinimal(self._ptr, array._ptr, &error.c_error)

        error.raise_message_not_ok("ArrowArrayViewSetArray()", code)
        self._array_base = array._base
        self._device = device
        return self

    @property
    def storage_type_id(self):
        return self._ptr.storage_type

    @property
    def storage_type(self):
        cdef const char* type_str = ArrowTypeString(self._ptr.storage_type)
        if type_str != NULL:
            return type_str.decode('UTF-8')

    @property
    def layout(self):
        return CLayout(self, <uintptr_t>&self._ptr.layout)

    def __len__(self):
        return self._ptr.length

    @property
    def length(self):
        return len(self)

    @property
    def offset(self):
        return self._ptr.offset

    @property
    def null_count(self):
        if self._ptr.null_count != -1:
            return self._ptr.null_count

        cdef ArrowBufferType buffer_type = self._ptr.layout.buffer_type[0]
        cdef const uint8_t* validity_bits = self._ptr.buffer_views[0].data.as_uint8

        if buffer_type != NANOARROW_BUFFER_TYPE_VALIDITY:
            self._ptr.null_count = 0
        elif validity_bits == NULL:
            self._ptr.null_count = 0
        elif self._device is DEVICE_CPU:
            self._ptr.null_count = (
                self._ptr.length -
                ArrowBitCountSet(validity_bits, self.offset, self.length)
            )

        return self._ptr.null_count

    @property
    def n_children(self):
        return self._ptr.n_children

    def child(self, int64_t i):
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")

        cdef CArrayView child = CArrayView(
            self._base,
            <uintptr_t>self._ptr.children[i]
        )

        child._device = self._device
        return child

    @property
    def children(self):
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def n_buffers(self):
        return self.layout.n_buffers

    def buffer_type(self, int64_t i):
        if i < 0 or i >= self.n_buffers:
            raise IndexError(f"{i} out of range [0, {self.n_buffers}]")

        buffer_type = self._ptr.layout.buffer_type[i]
        if buffer_type == NANOARROW_BUFFER_TYPE_VALIDITY:
            return "validity"
        elif buffer_type == NANOARROW_BUFFER_TYPE_TYPE_ID:
            return "type_id"
        elif buffer_type == NANOARROW_BUFFER_TYPE_UNION_OFFSET:
            return "union_offset"
        elif buffer_type == NANOARROW_BUFFER_TYPE_DATA_OFFSET:
            return "data_offset"
        elif buffer_type == NANOARROW_BUFFER_TYPE_DATA:
            return "data"
        else:
            return "none"

    def buffer(self, int64_t i):
        if i < 0 or i >= self.n_buffers:
            raise IndexError(f"{i} out of range [0, {self.n_buffers}]")

        cdef ArrowBufferView* buffer_view = &(self._ptr.buffer_views[i])

        # Check the buffer size here because the error later is cryptic.
        # Buffer sizes are set to -1 when they are "unknown", so because of errors
        # in nanoarrow/C or because the array is on a non-CPU device, that -1 value
        # could leak its way here.
        if buffer_view.size_bytes < 0:
            raise RuntimeError(f"ArrowArrayView buffer {i} has size_bytes < 0")

        return CBufferView(
            self._array_base,
            <uintptr_t>buffer_view.data.data,
            buffer_view.size_bytes,
            self._ptr.layout.buffer_data_type[i],
            self._ptr.layout.element_size_bits[i],
            self._device
        )

    @property
    def buffers(self):
        for i in range(self.n_buffers):
            yield self.buffer(i)

    @property
    def dictionary(self):
        if self._ptr.dictionary == NULL:
            return None
        else:
            return CArrayView(
                self,
                <uintptr_t>self._ptr.dictionary
            )

    def __repr__(self):
        return _repr_utils.array_view_repr(self)

    @staticmethod
    def from_schema(CSchema schema):
        cdef ArrowArrayView* c_array_view
        base = alloc_c_array_view(&c_array_view)

        cdef Error error = Error()
        cdef int code = ArrowArrayViewInitFromSchema(c_array_view,
                                                     schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayViewInitFromSchema()", code)

        return CArrayView(base, <uintptr_t>c_array_view)

    @staticmethod
    def from_array(CArray array, Device device=DEVICE_CPU):
        out = CArrayView.from_schema(array._schema)
        return out._set_array(array, device)


cdef class CArrayBuilder:
    cdef CArray c_array
    cdef ArrowArray* _ptr
    cdef bint _can_validate

    def __cinit__(self, CArray array):
        self.c_array = array
        self._ptr = array._ptr
        self._can_validate = True

    @staticmethod
    def allocate():
        return CArrayBuilder(CArray.allocate(CSchema.allocate()))

    def is_empty(self):
        if self._ptr.release == NULL:
            raise RuntimeError("CArrayBuilder is not initialized")

        return self._ptr.length == 0

    def init_from_type(self, int type_id):
        if self._ptr.release != NULL:
            raise RuntimeError("CArrayBuilder is already initialized")

        cdef int code = ArrowArrayInitFromType(self._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowArrayInitFromType()", code)

        code = ArrowSchemaInitFromType(self.c_array._schema._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowSchemaInitFromType()", code)

        return self

    def init_from_schema(self, CSchema schema):
        if self._ptr.release != NULL:
            raise RuntimeError("CArrayBuilder is already initialized")

        cdef Error error = Error()
        cdef int code = ArrowArrayInitFromSchema(self._ptr, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayInitFromType()", code)

        self.c_array._schema = schema
        return self

    def start_appending(self):
        cdef int code = ArrowArrayStartAppending(self._ptr)
        Error.raise_error_not_ok("ArrowArrayStartAppending()", code)
        return self

    def append_strings(self, obj):
        cdef int code
        cdef Py_ssize_t item_utf8_size
        cdef ArrowStringView item

        for py_item in obj:
            if py_item is None:
                code = ArrowArrayAppendNull(self._ptr, 1)
            else:
                # Cython raises the error from PyUnicode_AsUTF8AndSize()
                # in the event that py_item is not a str(); however, we
                # set item_utf8_size = 0 to be safe.
                item_utf8_size = 0
                item.data = PyUnicode_AsUTF8AndSize(py_item, &item_utf8_size)
                item.size_bytes = item_utf8_size
                code = ArrowArrayAppendString(self._ptr, item)

            if code != NANOARROW_OK:
                Error.raise_error(f"append string item {py_item}", code)

        return self

    def append_bytes(self, obj):
        cdef Py_buffer buffer
        cdef ArrowBufferView item

        for py_item in obj:
            if py_item is None:
                code = ArrowArrayAppendNull(self._ptr, 1)
            else:
                PyObject_GetBuffer(py_item, &buffer, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT)

                if buffer.ndim != 1:
                    raise ValueError("Can't append buffer with dimensions != 1 to binary array")

                if buffer.itemsize != 1:
                    PyBuffer_Release(&buffer)
                    raise ValueError("Can't append buffer with itemsize != 1 to binary array")

                item.data.data = buffer.buf
                item.size_bytes = buffer.len
                code = ArrowArrayAppendBytes(self._ptr, item)
                PyBuffer_Release(&buffer)

            if code != NANOARROW_OK:
                Error.raise_error(f"append bytes item {py_item}", code)

    def set_offset(self, int64_t offset):
        self.c_array._assert_valid()
        self._ptr.offset = offset
        return self

    def set_length(self, int64_t length):
        self.c_array._assert_valid()
        self._ptr.length = length
        return self

    def set_null_count(self, int64_t null_count):
        self.c_array._assert_valid()
        self._ptr.null_count = null_count
        return self

    def resolve_null_count(self):
        self.c_array._assert_valid()

        # This doesn't apply to unions. We currently don't have a schema view
        # or array view we can use to query the type ID, so just use the format
        # string for now.
        format = self.c_array.schema.format
        if format.startswith("+us:") or format.startswith("+ud:"):
            return self

        # Don't overwrite an explicit null count
        if self._ptr.null_count != -1:
            return self

        cdef ArrowBuffer* validity_buffer = ArrowArrayBuffer(self._ptr, 0)
        if validity_buffer.size_bytes == 0:
            self._ptr.null_count = 0
            return self

        # From _ArrowBytesForBits(), which is not included in nanoarrow_c.pxd
        # because it's an internal inline function.
        cdef int64_t bits = self._ptr.offset + self._ptr.length
        cdef int64_t bytes_required = (bits >> 3) + ((bits & 7) != 0)

        if validity_buffer.size_bytes < bytes_required:
            raise ValueError(
                f"Expected validity bitmap >= {bytes_required} bytes "
                f"but got validity bitmap with {validity_buffer.size_bytes} bytes"
            )

        cdef int64_t count = ArrowBitCountSet(
            validity_buffer.data,
            self._ptr.offset,
            self._ptr.length
        )
        self._ptr.null_count = self._ptr.length - count
        return self

    def set_buffer(self, int64_t i, CBuffer buffer, move=False):
        """Sets a buffer of this ArrowArray such the pointer at array->buffers[i] is
        equal to buffer->data and such that the buffer's lifcycle is managed by
        the array. If move is True, the input Python object that previously wrapped
        the ArrowBuffer will be invalidated, which is usually the desired behaviour
        if you built or imported a buffer specifically to build this array. If move
        is False (the default), this function will a make a shallow copy via another
        layer of Python object wrapping."""
        if i < 0 or i > 3:
            raise IndexError("i must be >= 0 and <= 3")

        self.c_array._assert_valid()
        if not move:
            buffer = CBuffer.from_pybuffer(buffer)

        ArrowBufferMove(buffer._ptr, ArrowArrayBuffer(self._ptr, i))

        # The buffer's lifecycle is now owned by the array; however, we need
        # array->buffers[i] to be updated such that it equals
        # ArrowArrayBuffer(array, i)->data.
        self._ptr.buffers[i] = ArrowArrayBuffer(self._ptr, i).data

        return self

    def set_child(self, int64_t i, CArray c_array, move=False):
        cdef CArray child = self.c_array.child(i)
        if child._ptr.release != NULL:
            ArrowArrayRelease(child._ptr)

        if not move:
            c_array_shallow_copy(c_array._base, c_array._ptr, child._ptr)
        else:
            ArrowArrayMove(c_array._ptr, child._ptr)

        # After setting children, we can't use the built-in validation done by
        # ArrowArrayFinishBuilding() because it assumes that the private_data of
        # each array (recursively) is one that was initialized by ArrowArrayInit()
        self._can_validate = False

        return self

    def finish(self, validation_level=None):
        self.c_array._assert_valid()
        cdef ArrowValidationLevel c_validation_level
        cdef Error error = Error()
        cdef int code

        if self._can_validate:
            c_validation_level = NANOARROW_VALIDATION_LEVEL_DEFAULT
            if validation_level == "full":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_FULL
            elif validation_level == "minimal":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_MINIMAL
            elif validation_level == "none":
                c_validation_level = NANOARROW_VALIDATION_LEVEL_NONE

            code = ArrowArrayFinishBuilding(self._ptr, c_validation_level, &error.c_error)
            error.raise_message_not_ok("ArrowArrayFinishBuildingDefault()", code)

        elif validation_level not in (None, "none"):
            raise NotImplementedError("Validation for array with children is not implemented")

        out = self.c_array
        self.c_array = CArray.allocate(CSchema.allocate())
        self._ptr = self.c_array._ptr
        self._can_validate = True

        return out


cdef class CArrayStream:
    """Low-level ArrowArrayStream wrapper

    This object is a literal wrapper around an ArrowArrayStream. It provides methods that
    that wrap the underlying C callbacks and handles the C Data interface lifecycle
    (i.e., initialized ArrowArrayStream structures are always released).

    See `nanoarrow.c_array_stream()` for construction and usage examples.
    """
    cdef object _base
    cdef ArrowArrayStream* _ptr
    cdef object _cached_schema

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowArrayStream*>addr
        self._cached_schema = None

    @staticmethod
    def allocate():
        cdef ArrowArrayStream* c_array_stream_out
        base = alloc_c_array_stream(&c_array_stream_out)
        return CArrayStream(base, <uintptr_t>c_array_stream_out)

    @staticmethod
    def from_c_arrays(arrays, CSchema schema, move=False, validate=True):
        cdef ArrowArrayStream* c_array_stream_out
        base = alloc_c_array_stream(&c_array_stream_out)

        # Don't create more copies than we have to (but make sure
        # one exists for validation if requested)
        cdef CSchema out_schema = schema
        if validate and not move:
            validate_schema = schema
            out_schema = schema.__deepcopy__()
        elif validate:
            validate_schema = schema.__deepcopy__()
            out_schema = schema
        elif not move:
            out_schema = schema.__deepcopy__()

        cdef int code = ArrowBasicArrayStreamInit(c_array_stream_out, out_schema._ptr, len(arrays))
        Error.raise_error_not_ok("ArrowBasicArrayStreamInit()", code)

        cdef ArrowArray tmp
        cdef CArray array
        for i in range(len(arrays)):
            array = arrays[i]

            if validate:
                assert_type_equal(array.schema, validate_schema, False)

            if not move:
                c_array_shallow_copy(array._base, array._ptr, &tmp)
                ArrowBasicArrayStreamSetArray(c_array_stream_out, i, &tmp)
            else:
                ArrowBasicArrayStreamSetArray(c_array_stream_out, i, array._ptr)

        cdef Error error = Error()
        if validate:
            code = ArrowBasicArrayStreamValidate(c_array_stream_out, &error.c_error)
            error.raise_message_not_ok("ArrowBasicArrayStreamValidate()", code)

        return CArrayStream(base, <uintptr_t>c_array_stream_out)

    def release(self):
        if self.is_valid():
            self._ptr.release(self._ptr)

    @staticmethod
    def _import_from_c_capsule(stream_capsule):
        """
        Import from a ArrowArrayStream PyCapsule.

        Parameters
        ----------
        stream_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_array_stream' containing an
            ArrowArrayStream pointer.
        """
        return CArrayStream(
            stream_capsule,
            <uintptr_t>PyCapsule_GetPointer(stream_capsule, 'arrow_array_stream')
        )

    def __arrow_c_stream__(self, requested_schema=None):
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
        self._assert_valid()

        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        cdef:
            ArrowArrayStream* c_array_stream_out

        array_stream_capsule = alloc_c_array_stream(&c_array_stream_out)
        ArrowArrayStreamMove(self._ptr, c_array_stream_out)
        return array_stream_capsule

    def _addr(self):
        return <uintptr_t>self._ptr

    def is_valid(self):
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("array stream pointer is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("array stream is released")

    def _get_schema(self, CSchema schema):
        self._assert_valid()
        cdef Error error = Error()
        cdef int code = ArrowArrayStreamGetSchema(self._ptr, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayStream::get_schema()", code)

    def _get_cached_schema(self):
        if self._cached_schema is None:
            self._cached_schema = CSchema.allocate()
            self._get_schema(self._cached_schema)

        return self._cached_schema

    def get_schema(self):
        """Get the schema associated with this stream
        """
        out = CSchema.allocate()
        self._get_schema(out)
        return out

    def get_next(self):
        """Get the next Array from this stream

        Raises StopIteration when there are no more arrays in this stream.
        """
        self._assert_valid()

        # We return a reference to the same Python object for each
        # Array that is returned. This is independent of get_schema(),
        # which is guaranteed to call the C object's callback and
        # faithfully pass on the returned value.

        cdef Error error = Error()
        cdef CArray array = CArray.allocate(self._get_cached_schema())
        cdef int code = ArrowArrayStreamGetNext(self._ptr, array._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowArrayStream::get_next()", code)

        if not array.is_valid():
            raise StopIteration()
        else:
            return array

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.release()

    def __repr__(self):
        return _repr_utils.array_stream_repr(self)


cdef class CMaterializedArrayStream:
    cdef CSchema _schema
    cdef CBuffer _array_ends
    cdef list _arrays
    cdef int64_t _total_length

    def __cinit__(self):
        self._arrays = []
        self._total_length = 0
        self._schema = CSchema.allocate()
        self._array_ends = CBuffer.empty()
        cdef int code = ArrowBufferAppendInt64(self._array_ends._ptr, 0)
        Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)

    cdef _finalize(self):
        self._array_ends._set_data_type(<ArrowType>_types.INT64)

    @property
    def schema(self):
        return self._schema

    def __getitem__(self, k):
        cdef int64_t kint
        cdef int array_i
        cdef const int64_t* sorted_offsets = <int64_t*>self._array_ends._ptr.data

        if isinstance(k, slice):
            raise NotImplementedError("index with slice")

        kint = k
        if kint < 0:
            kint += self._total_length
        if kint < 0 or kint >= self._total_length:
            raise IndexError(f"Index {kint} is out of range")

        array_i = ArrowResolveChunk64(kint, sorted_offsets, 0, len(self._arrays))
        kint -= sorted_offsets[array_i]
        return self._arrays[array_i], kint

    def __len__(self):
        return self._array_ends[len(self._arrays)]

    def __iter__(self):
        for c_array in self._arrays:
            for item_i in range(len(c_array)):
                yield c_array, item_i

    def array(self, int64_t i):
        return self._arrays[i]

    @property
    def n_arrays(self):
        return len(self._arrays)

    @property
    def arrays(self):
        return iter(self._arrays)

    def __arrow_c_stream__(self, requested_schema=None):
        # When an array stream from iterable is supported, that could be used here
        # to avoid unnessary shallow copies.
        stream = CArrayStream.from_c_arrays(
            self._arrays,
            self._schema,
            move=False,
            validate=False
        )

        return stream.__arrow_c_stream__(requested_schema=requested_schema)

    def child(self, int64_t i):
        cdef CMaterializedArrayStream out = CMaterializedArrayStream()
        cdef int code

        out._schema = self._schema.child(i)
        out._arrays = [chunk.child(i) for chunk in self._arrays]
        for child_chunk in out._arrays:
            out._total_length += len(child_chunk)
            code = ArrowBufferAppendInt64(out._array_ends._ptr, out._total_length)
            Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)

        out._finalize()
        return out

    @staticmethod
    def from_c_arrays(arrays, CSchema schema, bint validate=True):
        cdef CMaterializedArrayStream out = CMaterializedArrayStream()

        for array in arrays:
            if not isinstance(array, CArray):
                raise TypeError(f"Expected CArray but got {type(array).__name__}")

            if len(array) == 0:
                continue

            if validate:
                assert_type_equal(array.schema, schema, False)

            out._total_length += len(array)
            code = ArrowBufferAppendInt64(out._array_ends._ptr, out._total_length)
            Error.raise_error_not_ok("ArrowBufferAppendInt64()", code)
            out._arrays.append(array)

        out._schema = schema
        out._finalize()
        return out

    @staticmethod
    def from_c_array(CArray array):
        return CMaterializedArrayStream.from_c_arrays(
            [array],
            array.schema,
            validate=False
        )

    @staticmethod
    def from_c_array_stream(CArrayStream stream):
        with stream:
            return CMaterializedArrayStream.from_c_arrays(
                stream,
                stream._get_cached_schema(),
                validate=False
            )


cdef class CDeviceArray:
    cdef object _base
    cdef ArrowDeviceArray* _ptr
    cdef CSchema _schema

    def __cinit__(self, object base, uintptr_t addr, CSchema schema):
        self._base = base
        self._ptr = <ArrowDeviceArray*>addr
        self._schema = schema

    @staticmethod
    def _init_from_array(Device device, uintptr_t array_addr, CSchema schema):
        cdef ArrowArray* array_ptr = <ArrowArray*>array_addr
        cdef ArrowDeviceArray* device_array_ptr
        cdef void* sync_event = NULL
        holder = alloc_c_device_array(&device_array_ptr)
        cdef int code = ArrowDeviceArrayInit(device._ptr, device_array_ptr, array_ptr, sync_event)
        Error.raise_error_not_ok("ArrowDeviceArrayInit", code)

        return CDeviceArray(holder, <uintptr_t>device_array_ptr, schema)

    @property
    def schema(self):
        return self._schema

    @property
    def device_type(self):
        return DeviceType(self._ptr.device_type)

    @property
    def device_type_id(self):
        return self._ptr.device_type

    @property
    def device_id(self):
        return self._ptr.device_id

    @property
    def array(self):
        # TODO: We lose access to the sync_event here, so we probably need to
        # synchronize (or propagate it, or somehow prevent data access downstream)
        cdef CArray array = CArray(self, <uintptr_t>&self._ptr.array, self._schema)
        array._set_device(self._ptr.device_type, self._ptr.device_id)
        return array

    def view(self):
        return self.array.view()

    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema=requested_schema)

    def __arrow_c_device_array__(self, requested_schema=None):
        if requested_schema is not None:
            raise NotImplementedError("requested_schema")

        # TODO: evaluate whether we need to synchronize here or whether we should
        # move device arrays instead of shallow-copying them
        cdef ArrowDeviceArray* c_array_out
        device_array_capsule = alloc_c_device_array(&c_array_out)
        c_device_array_shallow_copy(self._base, self._ptr, c_array_out)

        return self._schema.__arrow_c_schema__(), device_array_capsule

    @staticmethod
    def _import_from_c_capsule(schema_capsule, device_array_capsule):
        """
        Import from an ArrowSchema and ArrowArray PyCapsule tuple.

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        device_array_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_device_array' containing an
            ArrowDeviceArray pointer.
        """
        cdef:
            CSchema out_schema
            CDeviceArray out

        out_schema = CSchema._import_from_c_capsule(schema_capsule)
        out = CDeviceArray(
            device_array_capsule,
            <uintptr_t>PyCapsule_GetPointer(device_array_capsule, 'arrow_device_array'),
            out_schema
        )

        return out

    def __repr__(self):
        return _repr_utils.device_array_repr(self)
