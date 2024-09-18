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

from nanoarrow._buffer import CBuffer, CBufferBuilder
from nanoarrow._utils import obj_is_buffer
from nanoarrow.c_schema import c_schema_view

from nanoarrow import _types


def c_buffer(obj, schema=None) -> CBuffer:
    """Owning, read-only ArrowBuffer wrapper

    If obj implements the Python buffer protocol, ``c_buffer()`` wraps
    obj in nanoarrow's owning buffer structure, the ArrowBuffer,
    such that it can be used to construct arrays. The ownership of the
    underlying buffer is handled by the Python buffer protocol
    (i.e., ``PyObject_GetBuffer()`` and ``PyBuffer_Release()``).

    If obj is iterable, a buffer will be allocated and populated with
    the contents of obj according to ``schema``. The
    ``schema`` parameter is required to create a buffer from
    a Python iterable. The ``struct`` module is currently used to encode
    values from obj into binary form.

    Unlike with :func:`c_array`, ``schema`` is explicitly
    honoured (or an error will be raised).

    Parameters
    ----------

    obj : buffer-like, tensor, or iterable
        A Python object that supports the Python buffer or DLPack protocols.
        This includes bytes, memoryview, bytearray, bulit-in types as well
        as numpy arrays.
    schema :  schema-like, optional
        The data type of the desired buffer as sanitized by
        :func:`c_schema`. Only values that make sense as buffer types are
        allowed (e.g., integer types, floating-point types, interval types,
        decimal types, binary, string, fixed-size binary).

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.c_buffer(b"1234")
    nanoarrow.c_buffer.CBuffer(uint8[4 b] 49 50 51 52)
    >>> na.c_buffer([1, 2, 3], na.int32())
    nanoarrow.c_buffer.CBuffer(int32[12 b] 1 2 3)
    """
    if isinstance(obj, CBuffer) and schema is None:
        return obj

    if obj_is_buffer(obj):
        if schema is not None:
            raise NotImplementedError(
                "c_buffer() with schema for pybuffer is not implemented"
            )
        return CBuffer.from_pybuffer(obj)

    if _obj_is_tensor(obj):
        if schema is not None:
            raise NotImplementedError(
                "c_buffer() with schema for DLPack is not implemented"
            )
        return CBuffer.from_dlpack(obj)

    if _obj_is_iterable(obj):
        buffer, _ = _c_buffer_from_iterable(obj, schema)
        return buffer

    raise TypeError(
        f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_buffer"
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
    if (
        schema_view.extension_name is not None
        or schema_view.storage_type_id != schema_view.type_id
    ):
        raise ValueError(
            f"Can't create buffer from iterable for type {schema_view.type}"
        )

    builder = CBufferBuilder()

    if schema_view.storage_type_id == _types.FIXED_SIZE_BINARY:
        builder.set_data_type(_types.BINARY, schema_view.fixed_size * 8)
    else:
        builder.set_data_type(schema_view.storage_type_id)

    # If we are using a typecode supported by the array module, it has much
    # faster implementations of safely building buffers from iterables
    if builder.format in array_typecodes and schema_view.storage_type_id != _types.BOOL:
        buf = array.array(builder.format, obj)
        return CBuffer.from_pybuffer(buf), len(buf)

    n_values = builder.write_elements(obj)
    return builder.finish(), n_values


def _obj_is_iterable(obj):
    return hasattr(obj, "__iter__")


def _obj_is_tensor(obj):
    return hasattr(obj, "__dlpack__")
