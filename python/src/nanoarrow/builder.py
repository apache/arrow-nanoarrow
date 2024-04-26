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

from typing import Any, Iterable, Literal, Tuple

from nanoarrow._lib import (
    CArray,
    CArrayBuilder,
    CArrowType,
    CBuffer,
    CBufferBuilder,
    CSchema,
    CSchemaBuilder,
    NoneAwareWrapperIterator,
    _obj_is_buffer,
)
from nanoarrow.c_lib import c_array, c_buffer, c_schema, c_schema_view


def c_array_from_buffers(
    schema,
    length: int,
    buffers: Iterable[Any],
    null_count: int = -1,
    offset: int = 0,
    children: Iterable[Any] = (),
    validation_level: Literal[None, "full", "default", "minimal", "none"] = None,
    move: bool = False,
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

    Examples
    --------

    >>> import nanoarrow as na
    >>> c_array = na.c_array_from_buffers(na.uint8(), 5, [None, b"12345"])
    >>> na.c_array_view(c_array)
    <nanoarrow.c_lib.CArrayView>
    - storage_type: 'uint8'
    - length: 5
    - offset: 0
    - null_count: 0
    - buffers[2]:
      - validity <bool[0 b] >
      - data <uint8[5 b] 49 50 51 52 53>
    - dictionary: NULL
    - children[0]:
    """
    schema = c_schema(schema)
    builder = CArrayBuilder.allocate()

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
        move = move or not isinstance(child_src, CArray)
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

    # Validate + finish
    return builder.finish(validation_level=validation_level)


def _c_buffer_from_iterable(obj, schema=None) -> CBuffer:
    import array

    # array.typecodes is not available in all PyPy versions.
    # Rather than guess, just don't use the array constructor if
    # this attribute is not available.
    if hasattr(array, "typecodes"):
        array_typecodes = array.typecodes
    else:
        array_typecodes = []

    if schema is None:
        raise ValueError("CBuffer from iterable requires schema")

    schema_view = c_schema_view(schema)
    if schema_view.storage_type_id != schema_view.type_id:
        raise ValueError(
            f"Can't create buffer from iterable for type {schema_view.type}"
        )

    builder = CBufferBuilder()

    if schema_view.storage_type_id == CArrowType.FIXED_SIZE_BINARY:
        builder.set_data_type(CArrowType.BINARY, schema_view.fixed_size * 8)
    else:
        builder.set_data_type(schema_view.storage_type_id)

    # If we are using a typecode supported by the array module, it has much
    # faster implementations of safely building buffers from iterables
    if (
        builder.format in array_typecodes
        and schema_view.storage_type_id != CArrowType.BOOL
    ):
        buf = array.array(builder.format, obj)
        return CBuffer.from_pybuffer(buf), len(buf)

    n_values = builder.write_elements(obj)
    return builder.finish(), n_values


class ArrayBuilder:
    @classmethod
    def infer_schema(cls, obj) -> Tuple[CSchema, Any]:
        raise NotImplementedError()

    def __init__(self, schema, *, _schema_view=None):
        self._schema = c_schema(schema)

        if _schema_view is not None:
            self._schema_view = _schema_view
        else:
            self._schema_view = c_schema_view(schema)

    def build_c_array(self, obj):
        builder = CArrayBuilder.allocate()
        builder.init_from_schema(self._schema)
        self.start_building(builder)
        self.append(builder, obj)
        return self.finish_building(builder)

    def start_building(self, c_builder: CArrayBuilder) -> None:
        pass

    def append(self, c_builder: CArrayBuilder, obj: Any) -> None:
        raise NotImplementedError()

    def finish_building(self, c_builder: CArrayBuilder) -> CArray:
        return c_builder.finish()


def _resolve_builder(obj) -> type[ArrayBuilder]:
    if _obj_is_empty(obj):
        return EmptyArrayBuilder

    if _obj_is_buffer(obj):
        return ArrayFromPyBufferBuilder

    if _obj_is_iterable(obj):
        return ArrayFromIterableBuilder

    raise TypeError(
        f"Can't reolve ArrayBuilder for object of type {type(obj).__name__}"
    )


def build_c_array(obj, schema=None) -> CArray:
    builder_cls = _resolve_builder(obj)

    if schema is None:
        obj, schema = builder_cls.infer_schema(obj)
    else:
        schema = c_schema(schema)

    builder = builder_cls(schema)
    return builder.build_c_array(obj)


def _obj_is_iterable(obj):
    return hasattr(obj, "__iter__")


def _obj_is_empty(obj):
    return hasattr(obj, "__len__") and len(obj) == 0


class EmptyArrayBuilder(ArrayBuilder):
    @classmethod
    def infer_schema(cls, obj) -> Tuple[Any, CSchema]:
        return obj, CSchemaBuilder.allocate().set_type(CArrowType.NA)

    def start_building(self, c_builder: CArrayBuilder) -> None:
        c_builder.start_appending()

    def append(self, c_builder: CArrayBuilder, obj: Any) -> None:
        if len(obj) != 0:
            raise ValueError(
                f"Can't build empty array from {type(obj).__name__} "
                f"with length {len(obj)}"
            )


class ArrayFromPyBufferBuilder(ArrayBuilder):
    @classmethod
    def infer_schema(cls, obj) -> Tuple[CBuffer, CSchema]:
        if not isinstance(obj, CBuffer):
            obj = CBuffer.from_pybuffer(obj)

        type_id = obj.data_type_id
        element_size_bits = obj.element_size_bits

        # Fixed-size binary needs a schema
        if type_id == CArrowType.BINARY and element_size_bits != 0:
            schema = (
                CSchemaBuilder.allocate()
                .set_type_fixed_size(
                    CArrowType.FIXED_SIZE_BINARY, element_size_bits // 8
                )
                .finish()
            )
        elif type_id == CArrowType.STRING:
            schema = CSchemaBuilder.allocate().set_type(CArrowType.INT8).finish()
        elif type_id == CArrowType.BINARY:
            schema = CSchemaBuilder.allocate().set_type(CArrowType.UINT8).finish()
        else:
            schema = CSchemaBuilder.allocate().set_type(type_id).finish()

        return obj, schema

    def __init__(self, schema, *, move: bool = False, _schema_view=None):
        super().__init__(schema, _schema_view=_schema_view)
        self._move = move

        if self._schema_view.buffer_format is None:
            raise ValueError(
                f"Can't build array of type {self._schema_view.type} from PyBuffer"
            )

    def append(self, c_builder: CArrayBuilder, obj: Any) -> None:
        if not c_builder.empty():
            raise ValueError("Can't append to non-empty ArrayFromPyBufferBuilder")

        if not isinstance(obj, CBuffer):
            obj = CBuffer.from_pybuffer(obj)

        if (
            self._schema_view.buffer_format in ("b", "c")
            and obj.format not in ("b", "c")
        ) and self._schema_view.buffer_format != obj.format:
            raise ValueError(
                f"Expected buffer with format '{self._schema_view.buffer_format}' "
                f"but got buffer with format '{obj.format}'"
            )

        c_builder.set_buffer(1, obj, move=self._move)
        c_builder.set_length(len(obj))
        c_builder.set_null_count(0)
        c_builder.set_offset(0)


class ArrayFromIterableBuilder(ArrayBuilder):
    @classmethod
    def infer_schema(cls, obj) -> Tuple[CBuffer, CSchema]:
        raise ValueError("schema is required to build array from iterable")

    def __init__(self, schema, *, _schema_view=None):
        super().__init__(schema, _schema_view=_schema_view)

        type_id = self._schema_view.type_id
        if type_id not in _ARRAY_BUILDER_FROM_ITERABLE_METHOD:
            raise ValueError(
                f"Can't build array of type {self._schema_view.type} from iterable"
            )

        method_name = _ARRAY_BUILDER_FROM_ITERABLE_METHOD[type_id]
        if method_name in _ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD:
            method_name = _ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD[method_name]

        self._append_impl = getattr(self, method_name)

    def start_building(self, c_builder: CArrayBuilder) -> None:
        c_builder.start_appending()

    def append(self, c_builder: CArrayBuilder, obj: Any) -> None:
        self._append_impl(c_builder, obj)

    def _append_strings(self, c_builder: CArrayBuilder, obj: Iterable) -> None:
        c_builder.append_strings(obj)

    def _append_bytes(self, c_builder: CArrayBuilder, obj: Iterable) -> None:
        c_builder.append_bytes(obj)

    def _build_nullable_array_using_array(
        self, c_builder: CArrayBuilder, obj: Iterable
    ) -> None:
        wrapper = NoneAwareWrapperIterator(
            obj, self._schema_view.storage_type_id, self._schema_view.fixed_size
        )
        self._append_using_array(c_builder, wrapper)

        _, null_count, validity = wrapper.finish()
        if validity is not None:
            c_builder.set_buffer(0, validity, move=True)

        c_builder.set_null_count(null_count)

    def _build_nullable_array_using_buffer_builder(
        self, c_builder: CArrayBuilder, obj: Iterable
    ) -> None:
        wrapper = NoneAwareWrapperIterator(
            obj, self._schema_view.storage_type_id, self._schema_view.fixed_size
        )
        self._append_using_buffer_builder(c_builder, wrapper)

        _, null_count, validity = wrapper.finish()
        if validity is not None:
            c_builder.set_buffer(0, validity, move=True)

        c_builder.set_null_count(null_count)

    def _append_using_array(self, c_builder: CArrayBuilder, obj: Iterable) -> None:
        from array import array

        py_array = array(self._schema_view.buffer_format, obj)
        buffer = CBuffer.from_pybuffer(py_array)
        c_builder.set_buffer(1, buffer, move=True)
        c_builder.set_length(len(buffer))
        c_builder.set_null_count(0)
        c_builder.set_offset(0)

    def _append_using_buffer_builder(
        self, c_builder: CArrayBuilder, obj: Iterable
    ) -> None:
        builder = CBufferBuilder()
        builder.set_data_type(self._schema_view.type_id)

        n_values = builder.write_elements(obj)

        buffer = builder.finish()
        c_builder.set_buffer(1, buffer, move=True)
        c_builder.set_length(n_values)
        c_builder.set_null_count(0)
        c_builder.set_offset(0)


_ARRAY_BUILDER_FROM_ITERABLE_METHOD = {
    CArrowType.BOOL: "_append_using_buffer_builder",
    CArrowType.HALF_FLOAT: "_append_using_buffer_builder",
    CArrowType.INTERVAL_MONTH_DAY_NANO: "_append_using_buffer_builder",
    CArrowType.INTERVAL_DAY_TIME: "_append_using_buffer_builder",
    CArrowType.INTERVAL_MONTHS: "_append_using_buffer_builder",
    CArrowType.BINARY: "_append_bytes",
    CArrowType.LARGE_BINARY: "_append_bytes",
    CArrowType.FIXED_SIZE_BINARY: "_append_bytes",
    CArrowType.STRING: "_append_strings",
    CArrowType.LARGE_STRING: "_append_strings",
    CArrowType.INT8: "_append_using_array",
    CArrowType.UINT8: "_append_using_array",
    CArrowType.INT16: "_append_using_array",
    CArrowType.UINT16: "_append_using_array",
    CArrowType.INT32: "_append_using_array",
    CArrowType.UINT32: "_append_using_array",
    CArrowType.INT64: "_append_using_array",
    CArrowType.UINT64: "_append_using_array",
    CArrowType.FLOAT: "_append_using_array",
    CArrowType.DOUBLE: "_append_using_array",
}

_ARRAY_BUILDER_FROM_NULLABLE_ITERABLE_METHOD = {
    "_append_using_array": "_build_nullable_array_using_array",
    "_append_using_buffer_builder": "_build_nullable_array_using_buffer_builder",
}
