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

from typing import Iterable, Any, Literal

from nanoarrow._lib import (
    CArray,
    CArrayBuilder,
    CArrowType,
    CBuffer,
    CBufferBuilder,
    CSchemaBuilder,
    NoneAwareWrapperIterator,
)
from nanoarrow.c_lib import c_array, c_schema, c_buffer, c_schema_view


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


# Invokes the buffer protocol on obj
def _c_array_from_pybuffer(obj) -> CArray:
    buffer = CBuffer.from_pybuffer(obj)
    type_id = buffer.data_type_id
    element_size_bits = buffer.element_size_bits

    builder = CArrayBuilder.allocate()

    # Fixed-size binary needs a schema
    if type_id == CArrowType.BINARY and element_size_bits != 0:
        c_schema = (
            CSchemaBuilder.allocate()
            .set_type_fixed_size(CArrowType.FIXED_SIZE_BINARY, element_size_bits // 8)
            .finish()
        )
        builder.init_from_schema(c_schema)
    elif type_id == CArrowType.STRING:
        builder.init_from_type(int(CArrowType.INT8))
    elif type_id == CArrowType.BINARY:
        builder.init_from_type(int(CArrowType.UINT8))
    else:
        builder.init_from_type(int(type_id))

    # Set the length
    builder.set_length(len(buffer))

    # Move ownership of the ArrowBuffer wrapped by buffer to builder.buffer(1)
    builder.set_buffer(1, buffer)

    # No nulls or offset from a PyBuffer
    builder.set_null_count(0)
    builder.set_offset(0)

    return builder.finish()


def _c_array_from_iterable(obj, schema=None) -> CArray:
    if schema is None:
        raise ValueError("schema is required for CArray import from iterable")

    obj_len = -1
    if hasattr(obj, "__len__"):
        obj_len = len(obj)

    # We can always create an array from an empty iterable, even for types
    # not supported by _c_buffer_from_iterable()
    if obj_len == 0:
        builder = CArrayBuilder.allocate()
        builder.init_from_schema(schema)
        builder.start_appending()
        return builder.finish()

    # We need to know a few things about the data type to choose the appropriate
    # strategy for building the array.
    schema_view = c_schema_view(schema)

    if schema_view.storage_type_id != schema_view.type_id:
        raise ValueError(
            f"Can't create array from iterable for type {schema_view.type}"
        )

    # Handle variable-size binary types (string, binary)
    if schema_view.type_id in (CArrowType.STRING, CArrowType.LARGE_STRING):
        builder = CArrayBuilder.allocate()
        builder.init_from_schema(schema)
        builder.start_appending()
        builder.append_strings(obj)
        return builder.finish()
    elif schema_view.type_id in (CArrowType.BINARY, CArrowType.LARGE_BINARY):
        builder = CArrayBuilder.allocate()
        builder.init_from_schema(schema)
        builder.start_appending()
        builder.append_bytes(obj)
        return builder.finish()

    # Creating a buffer from an iterable does not handle None values,
    # but we can do so here with the NoneAwareWrapperIterator() wrapper.
    # This approach is quite a bit slower, so only do it for a nullable
    # type.
    if schema_view.nullable:
        obj_wrapper = NoneAwareWrapperIterator(
            obj, schema_view.storage_type_id, schema_view.fixed_size
        )

        if obj_len > 0:
            obj_wrapper.reserve(obj_len)

        data, _ = _c_buffer_from_iterable(obj_wrapper, schema_view)
        n_values, null_count, validity = obj_wrapper.finish()
    else:
        data, n_values = _c_buffer_from_iterable(obj, schema_view)
        null_count = 0
        validity = None

    return c_array_from_buffers(
        schema, n_values, buffers=(validity, data), null_count=null_count, move=True
    )


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
