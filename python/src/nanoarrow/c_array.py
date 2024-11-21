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

from typing import Any, Iterable, Literal, Tuple, Union

from nanoarrow._array import CArray, CArrayBuilder, CArrayView, CDeviceArray
from nanoarrow._buffer import CBuffer, CBufferBuilder, NoneAwareWrapperIterator
from nanoarrow._device import DEVICE_CPU, Device
from nanoarrow._schema import CSchema, CSchemaBuilder
from nanoarrow._utils import obj_is_buffer, obj_is_capsule
from nanoarrow.c_buffer import c_buffer
from nanoarrow.c_schema import c_schema, c_schema_view
from nanoarrow.extension import resolve_extension

from nanoarrow import _types


def c_array(obj, schema=None) -> CArray:
    """ArrowArray wrapper

    This class provides a user-facing interface to access the fields of an ArrowArray
    as defined in the Arrow C Data interface, holding an optional reference to a
    :class:`CSchema` that can be used to safely deserialize the content.

    These objects are created using :func:`c_array`, which accepts any array-like
    object according to the Arrow PyCapsule interface, Python buffer protocol,
    or iterable of Python objects.

    This Python wrapper allows access to array fields but does not automatically
    deserialize their content: use :func:`c_array_view` to validate and deserialize
    the content into a more easily inspectable object.

    Note that the :class:`CArray` objects returned by ``.child()`` hold strong
    references to the original ``ArrowArray`` to avoid copies while inspecting an
    imported structure.

    Parameters
    ----------
    obj : array-like
        An object supporting the Arrow PyCapsule interface, the Python buffer
        protocol, or an iterable of Python objects.
    schema : schema-like or None
        A schema-like object as sanitized by :func:`c_schema` or None. This value
        will be used to request a data type from ``obj``; however, the conversion
        is best-effort (i.e., the data type of the returned ``CArray`` may be
        different than ``schema``).

    Examples
    --------

    >>> import nanoarrow as na
    >>> # Create from iterable
    >>> array = na.c_array([1, 2, 3], na.int32())
    >>> # Create from Python buffer (e.g., numpy array)
    >>> import numpy as np
    >>> array = na.c_array(np.array([1, 2, 3]))
    >>> # Create from Arrow PyCapsule (e.g., pyarrow array)
    >>> import pyarrow as pa
    >>> array = na.c_array(pa.array([1, 2, 3]))
    >>> # Access array fields
    >>> array.length
    3
    >>> array.null_count
    0
    """

    if schema is not None:
        schema = c_schema(schema)

    if isinstance(obj, CArray) and schema is None:
        return obj

    # Try Arrow PyCapsule protocol
    if hasattr(obj, "__arrow_c_array__"):
        schema_capsule = None if schema is None else schema.__arrow_c_schema__()
        return CArray._import_from_c_capsule(
            *obj.__arrow_c_array__(requested_schema=schema_capsule)
        )

    # Try import of bare capsule
    if obj_is_capsule(obj, "arrow_array"):
        if schema is None:
            schema_capsule = CSchema.allocate()._capsule
        else:
            schema_capsule = schema.__arrow_c_schema__()

        return CArray._import_from_c_capsule(schema_capsule, obj)

    # Try _export_to_c for Array/RecordBatch objects if pyarrow < 14.0
    if _obj_is_pyarrow_array(obj):
        out = CArray.allocate(CSchema.allocate())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out

    # Use the ArrayBuilder classes to handle various strategies for other
    # types of objects (e.g., iterable, pybuffer, empty).
    try:
        builder_cls = _resolve_builder(obj)
    except Exception as e:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} "
            f"to nanoarrow.c_array: \n {e}"
        ) from e

    try:
        if schema is None:
            obj, schema = builder_cls.infer_schema(obj)

        builder = builder_cls(schema)
        return builder.build_c_array(obj)
    except Exception as e:
        raise ValueError(
            f"An error occurred whilst converting {type(obj).__name__} "
            f"to nanoarrow.c_array: \n {e}"
        ) from e


def _resolve_builder(obj):
    if _obj_is_empty(obj):
        return EmptyArrayBuilder

    if obj_is_buffer(obj):
        return ArrayFromPyBufferBuilder

    if _obj_is_iterable(obj):
        return ArrayFromIterableBuilder

    raise TypeError(
        f"Can't resolve ArrayBuilder for object of type {type(obj).__name__}"
    )


def allocate_c_array(schema=None) -> CArray:
    """Allocate an uninitialized ArrowArray

    Examples
    --------

    >>> import pyarrow as pa
    >>> from nanoarrow.c_array import allocate_c_array
    >>> array = allocate_c_array()
    >>> pa.array([1, 2, 3])._export_to_c(array._addr())
    """
    if schema is not None:
        schema = c_schema(schema)

    return CArray.allocate(CSchema.allocate() if schema is None else schema)


def c_array_view(obj, schema=None) -> CArrayView:
    """ArrowArrayView wrapper

    The ``ArrowArrayView`` is a nanoarrow C library structure that provides
    structured access to buffers addresses, buffer sizes, and buffer
    data types. The buffer data is usually propagated from an ArrowArray
    but can also be propagated from other types of objects (e.g., serialized
    IPC). The offset and length of this view are independent of its parent
    (i.e., this object can also represent a slice of its parent).

    Examples
    --------

    >>> import pyarrow as pa
    >>> import numpy as np
    >>> import nanoarrow as na
    >>> from nanoarrow.c_array import c_array_view
    >>>
    >>> array = na.c_array(pa.array(["one", "two", "three", None]))
    >>> array_view = c_array_view(array)
    >>> np.array(array_view.buffer(1))
    array([ 0,  3,  6, 11, 11], dtype=int32)
    >>> np.array(array_view.buffer(2))
    array([b'o', b'n', b'e', b't', b'w', b'o', b't', b'h', b'r', b'e', b'e'],
          dtype='|S1')
    """

    if isinstance(obj, CArrayView) and schema is None:
        return obj

    return c_array(obj, schema).view()


def c_array_from_buffers(
    schema,
    length: int,
    buffers: Iterable[Any],
    null_count: int = -1,
    offset: int = 0,
    children: Iterable[Any] = (),
    validation_level: Literal[None, "full", "default", "minimal", "none"] = None,
    move: bool = False,
    device: Union[Device, None] = None,
) -> CArray:
    """Create an ArrowArray wrapper from components

    Given a schema, build an ArrowArray buffer-wise. This allows almost any array
    to be assembled; however, requires some knowledge of the Arrow Columnar
    specification. This function will do its best to validate the sizes and
    content of buffers according to ``validation_level``; however, not all
    types of arrays can currently be validated when constructed in this way.

    Parameters
    ----------
    schema : schema-like
        The data type of the desired array as sanitized by :func:`c_schema`.
    length : int
        The length of the output array.
    buffers : Iterable of buffer-like or None
        An iterable of buffers as sanitized by :func:`c_buffer`. Any object
        supporting the Python Buffer protocol is accepted. Buffer data types
        are not checked. A buffer value of ``None`` will skip setting a buffer
        (i.e., that buffer will be of length zero and its pointer will
        be ``NULL``).
    null_count : int, optional
        The number of null values, if known in advance. If -1 (the default),
        the null count will be calculated based on the validity bitmap. If
        the validity bitmap was set to ``None``, the calculated null count
        will be zero.
    offset : int, optional
        The logical offset from the start of the array.
    children : Iterable of array-like
        An iterable of arrays used to set child fields of the array. Can contain
        any object accepted by :func:`c_array`. Must contain the exact number of
        required children as specifed by ``schema``.
    validation_level: None or str, optional
        One of "none" (no check), "minimal" (check buffer sizes that do not require
        dereferencing buffer content), "default" (check all buffer sizes), or "full"
        (check all buffer sizes and all buffer content). The default, ``None``,
        will validate at the "default" level where possible.
    move : bool, optional
        Use ``True`` to move ownership of any input buffers or children to the
        output array.
    device : Device, optional
        An explicit device to use when constructing this array. If specified,
        this function will construct a :class:`CDeviceArray`; if unspecified,
        this function will construct a :class:`CArray` on the CPU device.

    Examples
    --------

    >>> import nanoarrow as na
    >>> c_array = na.c_array_from_buffers(na.uint8(), 5, [None, b"12345"])
    >>> na.Array(c_array).inspect()
    <ArrowArray uint8>
    - length: 5
    - offset: 0
    - null_count: 0
    - buffers[2]:
      - validity <bool[0 b] >
      - data <uint8[5 b] 49 50 51 52 53>
    - dictionary: NULL
    - children[0]:
    """
    if device is None:
        explicit_device = False
        device = DEVICE_CPU
    else:
        explicit_device = True

    schema = c_schema(schema)
    builder = CArrayBuilder.allocate(device)

    # Ensures that the output array->n_buffers is set and that the correct number
    # of children have been initialized.
    builder.init_from_schema(schema)

    # Set buffers, optionally moving ownership of the buffers as well (i.e.,
    # the objects in the input buffers would be replaced with an empty ArrowBuffer)
    for i, buffer in enumerate(buffers):
        if buffer is None:
            continue

        # If we're setting a CBuffer from something else, we can avoid an extra
        # level of Python wrapping by using move=True
        move = move or not isinstance(buffer, CBuffer)
        builder.set_buffer(i, c_buffer(buffer), move=move)

    # Set children, optionally moving ownership of the children as well (i.e.,
    # the objects in the input children would be marked released).
    n_children = 0
    for child_src in children:
        # If we're setting a CArray from something else, we can avoid an extra
        # level of Python wrapping by using move=True
        move = move or not isinstance(child_src, (CArray, CDeviceArray))
        if move and isinstance(child_src, CDeviceArray):
            child_src = child_src.array

        builder.set_child(n_children, c_array(child_src), move=move)
        n_children += 1

    if n_children != schema.n_children:
        raise ValueError(f"Expected {schema.n_children} children but got {n_children}")

    # Set array fields
    builder.set_length(length)
    builder.set_offset(offset)
    builder.set_null_count(null_count)

    # Calculates the null count if -1 (and if applicable)
    builder.resolve_null_count()

    # Validate + finish. If device is specified (even CPU), always
    # return a device array.
    if not explicit_device:
        return builder.finish(validation_level=validation_level)
    else:
        return builder.finish_device()


class ArrayBuilder:
    """Internal utility to build CArrays from various types of input

    This class and its subclasses are designed to help separate the code
    that actually builds a CArray from the code that chooses the strategy
    used to do the building.
    """

    @classmethod
    def infer_schema(cls, obj) -> Tuple[CSchema, Any]:
        """Infer the Arrow data type from a target object

        Returns the type as a :class:`CSchema` and an object that can be
        consumed in the same way by append() in the event it had to be
        modified to infer its type (e.g., for an iterable, it would be
        necessary to consume the first element from the original iterator).
        """
        raise NotImplementedError()

    def __init__(self, schema):
        self._schema = c_schema(schema)
        self._schema_view = c_schema_view(self._schema)
        self._c_builder = CArrayBuilder.allocate()
        self._c_builder.init_from_schema(self._schema)

    def build_c_array(self, obj):
        self.start_building()
        self.append(obj)
        return self.finish_building()

    def start_building(self) -> None:
        pass

    def append(self, obj: Any) -> None:
        raise NotImplementedError()

    def finish_building(self) -> CArray:
        return self._c_builder.finish()


class EmptyArrayBuilder(ArrayBuilder):
    """Build an empty CArray of any type

    This builder accepts any empty input and produces a valid length zero
    array as output.
    """

    @classmethod
    def infer_schema(cls, obj) -> Tuple[Any, CSchema]:
        return obj, CSchemaBuilder.allocate().set_type(_types.NA)

    def start_building(self) -> None:
        self._c_builder.start_appending()

    def append(self, obj: Any) -> None:
        if len(obj) != 0:
            raise ValueError(
                f"Can't build empty array from {type(obj).__name__} "
                f"with length {len(obj)}"
            )


class ArrayFromPyBufferBuilder(ArrayBuilder):
    """Build a CArray from a Python Buffer

    This builder converts a Python buffer (e.g., numpy array, bytes, array.array)
    to a CArray (without copying the contents of the buffer).
    """

    @classmethod
    def infer_schema(cls, obj) -> Tuple[CBuffer, CSchema]:
        if not isinstance(obj, CBuffer):
            obj = CBuffer.from_pybuffer(obj)

        type_id = obj.data_type_id
        element_size_bits = obj.element_size_bits

        # Fixed-size binary needs a schema
        if type_id == _types.BINARY and element_size_bits != 0:
            schema = (
                CSchemaBuilder.allocate()
                .set_type_fixed_size(_types.FIXED_SIZE_BINARY, element_size_bits // 8)
                .finish()
            )
        elif type_id == _types.STRING:
            schema = CSchemaBuilder.allocate().set_type(_types.INT8).finish()
        elif type_id == _types.BINARY:
            schema = CSchemaBuilder.allocate().set_type(_types.UINT8).finish()
        else:
            schema = CSchemaBuilder.allocate().set_type(type_id).finish()

        return obj, schema

    def __init__(self, schema):
        super().__init__(schema)

        ext = resolve_extension(self._schema_view)
        self._append_ext = None
        if ext is not None:
            self._append_ext = ext.get_buffer_appender(self._schema, self)
        elif self._schema_view.extension_name:
            raise NotImplementedError(
                "Can't create array for unregistered extension "
                f"'{self._schema_view.extension_name}'"
            )

        if self._schema_view.storage_buffer_format is None:
            raise ValueError(
                f"Can't build array of type {self._schema_view.type} from PyBuffer"
            )

    def append(self, obj: Any) -> None:
        if not self._c_builder.is_empty():
            raise ValueError("Can't append to non-empty ArrayFromPyBufferBuilder")

        if not isinstance(obj, CBuffer):
            obj = CBuffer.from_pybuffer(obj)

        if self._append_ext is not None:
            return self._append_ext(obj)

        return self._append_impl(obj)

    def _append_impl(self, obj):
        if (
            self._schema_view.buffer_format in ("b", "c")
            and obj.format not in ("b", "c")
        ) and self._schema_view.buffer_format != obj.format:
            raise ValueError(
                f"Expected buffer with format '{self._schema_view.buffer_format}' "
                f"but got buffer with format '{obj.format}'"
            )

        self._c_builder.set_buffer(1, obj)
        self._c_builder.set_length(len(obj))
        self._c_builder.set_null_count(0)
        self._c_builder.set_offset(0)


class ArrayFromIterableBuilder(ArrayBuilder):
    """Build a CArray from an iterable of scalar objects

    This builder converts an iterable to a CArray using some heuristics to pick
    the fastest available method for converting to a particular type of array.
    Briefly, the methods are (1) ArrowArrayAppendXXX() functions from the C
    library (string, binary), (2) array.array() (integer/float except float16),
    (3) CBufferBuilder.write_elements() (everything else).
    """

    @classmethod
    def infer_schema(cls, obj) -> Tuple[CBuffer, CSchema]:
        raise ValueError("schema is required to build array from iterable")

    def __init__(self, schema):
        super().__init__(schema)

        # Resolve the method name we are going to use to do the building from
        # the provided schema.
        ext = resolve_extension(self._schema_view)
        if ext is not None:
            maybe_appender = ext.get_iterable_appender(self._schema, self)
            if maybe_appender:
                self._append_impl = maybe_appender
                return
        elif self._schema_view.extension_name:
            raise NotImplementedError(
                f"Can't create array for unregistered extension "
                f"'{self._schema_view.extension_name}'"
            )

        type_id = self._schema_view.type_id
        if type_id not in _ARRAY_BUILDER_FROM_ITERABLE_METHOD:
            raise ValueError(
                f"Can't build array of type {self._schema_view.type} from iterable"
            )

        method_name = _ARRAY_BUILDER_FROM_ITERABLE_METHOD[type_id]

        # If there might be nulls, we may need to pick a different strategy
        if (
            self._schema_view.nullable
            and method_name in _ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD
        ):
            method_name = _ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD[method_name]

        self._append_impl = getattr(self, method_name)

    def start_building(self) -> None:
        self._c_builder.start_appending()

    def append(self, obj: Any) -> None:
        self._append_impl(obj)

    def _append_strings(self, obj: Iterable) -> None:
        self._c_builder.append_strings(obj)

    def _append_bytes(self, obj: Iterable) -> None:
        self._c_builder.append_bytes(obj)

    def _build_nullable_array_using_array(self, obj: Iterable) -> None:
        wrapper = NoneAwareWrapperIterator(
            obj, self._schema_view.storage_type_id, self._schema_view.fixed_size
        )
        self._append_using_array(wrapper)

        _, null_count, validity = wrapper.finish()
        if validity is not None:
            self._c_builder.set_buffer(0, validity, move=True)

        self._c_builder.set_null_count(null_count)

    def _build_nullable_array_using_buffer_builder(self, obj: Iterable) -> None:
        wrapper = NoneAwareWrapperIterator(
            obj, self._schema_view.storage_type_id, self._schema_view.fixed_size
        )
        self._append_using_buffer_builder(wrapper)

        _, null_count, validity = wrapper.finish()
        if validity is not None:
            self._c_builder.set_buffer(0, validity, move=True)

        self._c_builder.set_null_count(null_count)

    def _append_using_array(self, obj: Iterable) -> None:
        from array import array

        py_array = array(self._schema_view.storage_buffer_format, obj)
        buffer = CBuffer.from_pybuffer(py_array)
        self._c_builder.set_buffer(1, buffer, move=True)
        self._c_builder.set_length(len(buffer))
        self._c_builder.set_null_count(0)
        self._c_builder.set_offset(0)

    def _append_using_buffer_builder(self, obj: Iterable) -> None:
        builder = CBufferBuilder()
        builder.set_data_type(self._schema_view.type_id)

        n_values = builder.write_elements(obj)

        buffer = builder.finish()
        self._c_builder.set_buffer(1, buffer, move=True)
        self._c_builder.set_length(n_values)
        self._c_builder.set_null_count(0)
        self._c_builder.set_offset(0)


_ARRAY_BUILDER_FROM_ITERABLE_METHOD = {
    _types.BOOL: "_append_using_buffer_builder",
    _types.HALF_FLOAT: "_append_using_buffer_builder",
    _types.INTERVAL_MONTH_DAY_NANO: "_append_using_buffer_builder",
    _types.INTERVAL_DAY_TIME: "_append_using_buffer_builder",
    _types.INTERVAL_MONTHS: "_append_using_buffer_builder",
    _types.BINARY: "_append_bytes",
    _types.LARGE_BINARY: "_append_bytes",
    _types.FIXED_SIZE_BINARY: "_append_bytes",
    _types.BINARY_VIEW: "_append_bytes",
    _types.STRING: "_append_strings",
    _types.LARGE_STRING: "_append_strings",
    _types.STRING_VIEW: "_append_strings",
    _types.INT8: "_append_using_array",
    _types.UINT8: "_append_using_array",
    _types.INT16: "_append_using_array",
    _types.UINT16: "_append_using_array",
    _types.INT32: "_append_using_array",
    _types.UINT32: "_append_using_array",
    _types.INT64: "_append_using_array",
    _types.UINT64: "_append_using_array",
    _types.FLOAT: "_append_using_array",
    _types.DOUBLE: "_append_using_array",
    _types.TIMESTAMP: "_append_using_array",
    _types.DATE32: "_append_using_array",
    _types.DATE64: "_append_using_array",
    _types.DURATION: "_append_using_array",
}

_ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD = {
    "_append_using_array": "_build_nullable_array_using_array",
    "_append_using_buffer_builder": "_build_nullable_array_using_buffer_builder",
}


def _obj_is_iterable(obj):
    return hasattr(obj, "__iter__")


def _obj_is_empty(obj):
    return hasattr(obj, "__len__") and len(obj) == 0


# This is a heuristic for detecting a pyarrow.Array or pyarrow.RecordBatch
# for pyarrow < 14.0.0, after which the the __arrow_c_array__ protocol
# is sufficient to detect such an array. This check can't use isinstance()
# to avoid importing pyarrow unnecessarily.
def _obj_is_pyarrow_array(obj):
    obj_type = type(obj)
    if not obj_type.__module__.startswith("pyarrow"):
        return False

    if not obj_type.__name__.endswith("Array") and obj_type.__name__ != "RecordBatch":
        return False

    return hasattr(obj, "_export_to_c")
