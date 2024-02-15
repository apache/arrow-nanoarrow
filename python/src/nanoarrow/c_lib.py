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

"""Arrow and nanoarrow C structure wrappers

These classes and their constructors wrap Arrow C Data/Stream interface structures
(i.e., ``ArrowArray``, ``ArrowSchema``, and ``ArrowArrayStream``) and the
nanoarrow C library structures that help deserialize their content (i.e., the
``ArrowSchemaView`` and ``ArrowArrayView``). These wrappers are currently implemented
in Cython and their scope is limited to lifecycle management and member access as
Python objects.
"""

from typing import Any, Iterable, Literal

from nanoarrow._lib import (
    CArray,
    CArrayBuilder,
    CArrayStream,
    CArrayView,
    CArrowType,
    CBuffer,
    CBufferBuilder,
    CSchema,
    CSchemaBuilder,
    CSchemaView,
    _obj_is_buffer,
    _obj_is_capsule,
)


def c_schema(obj=None) -> CSchema:
    """ArrowSchema wrapper

    The ``CSchema`` class provides a Python-friendly interface to access the fields
    of an ``ArrowSchema`` as defined in the Arrow C Data interface. These objects
    are created using `nanoarrow.c_schema()`, which accepts any schema or
    data type-like object according to the Arrow PyCapsule interface.

    This Python wrapper allows access to schema struct members but does not
    automatically deserialize their content: use :func:`c_schema_view` to validate
    and deserialize the content into a more easily inspectable object.

    Note that the :class:`CSchema` objects returned by ``.child()`` hold strong
    references to the original `ArrowSchema` to avoid copies while inspecting an
    imported structure.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.c_schema(pa.int32())
    >>> schema.is_valid()
    True
    >>> schema.format
    'i'
    >>> schema.name
    ''
    """

    if isinstance(obj, CSchema):
        return obj

    if hasattr(obj, "__arrow_c_schema__"):
        return CSchema._import_from_c_capsule(obj.__arrow_c_schema__())

    if _obj_is_capsule(obj, "arrow_schema"):
        return CSchema._import_from_c_capsule(obj)

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CSchema.allocate()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_schema"
        )


def c_array(obj, requested_schema=None) -> CArray:
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
    requested_schema : schema-like or None
        A schema-like object as sanitized by :func:`c_schema` or None. This value
        will be used to request a data type from ``obj``; however, the conversion
        is best-effort (i.e., the data type of the returned ``CArray`` may be
        different than ``requested_schema``).

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

    if requested_schema is not None:
        requested_schema = c_schema(requested_schema)

    if isinstance(obj, CArray) and requested_schema is None:
        return obj

    # Try Arrow PyCapsule protocol
    if hasattr(obj, "__arrow_c_array__"):
        requested_schema_capsule = (
            None if requested_schema is None else requested_schema.__arrow_c_schema__()
        )
        return CArray._import_from_c_capsule(
            *obj.__arrow_c_array__(requested_schema=requested_schema_capsule)
        )

    # Try buffer protocol (e.g., numpy arrays or a c_buffer())
    if _obj_is_buffer(obj):
        return _c_array_from_pybuffer(obj)

    # Try import of bare capsule
    if _obj_is_capsule(obj, "arrow_array"):
        if requested_schema is None:
            requested_schema_capsule = CSchema.allocate()._capsule
        else:
            requested_schema_capsule = requested_schema.__arrow_c_schema__()

        return CArray._import_from_c_capsule(requested_schema_capsule, obj)

    # Try _export_to_c for Array/RecordBatch objects if pyarrow < 14.0
    if _obj_is_pyarrow_array(obj):
        out = CArray.allocate(CSchema.allocate())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out

    # Try import of iterable
    if _obj_is_iterable(obj):
        return _c_array_from_iterable(obj, requested_schema)

    raise TypeError(
        f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_array"
    )


def c_array_from_buffers(
    schema,
    length: int,
    buffers: Iterable[Any],
    null_count: int = -1,
    offset: int = 0,
    children: Iterable[Any] = (),
    validation_level: Literal["full", "default", "minimal", "none"] = "default",
) -> CArray:
    """Create an ArrowArray wrapper from components

    Given a schema, build an ArrowArray buffer-wise. This allows almost any array
    to be assembled; however, requires some knowledge of the Arrow Columnar
    specification. This function will do its best to validate the sizes and
    content of buffers according to ``validation_level``, which can be set
    to ``"full""`` for maximum safety.

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
    validation_level: str, optional
        One of "none" (no check), "minimal" (check buffer sizes that do not require
        dereferencing buffer content), "default" (check all buffer sizes), or "full"
        (check all buffer sizes and all buffer content).

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

    # This is slightly wasteful: it will allocate arrays recursively and we are about
    # to immediately release them and replace them with another value. We could also
    # create an ArrowArrayView from the buffers, which would make it more
    # straightforward to check the buffer types and avoid the extra structure
    # allocation.
    builder.init_from_schema(schema)

    # Set buffers. This moves ownership of the buffers as well (i.e., the objects
    # in the input buffers are replaced with an empty ArrowBuffer)
    for i, buffer in enumerate(buffers):
        if buffer is None:
            continue
        builder.set_buffer(i, c_buffer(buffer))

    # Set children. This moves ownership of the children as well (i.e., the objects
    # in the input children are invalidated).
    n_children = 0
    for child_src in children:
        builder.set_child(n_children, c_array(child_src))
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


def c_array_stream(obj=None, requested_schema=None) -> CArrayStream:
    """ArrowArrayStream wrapper

    This class provides a user-facing interface to access the fields of
    an ArrowArrayStream as defined in the Arrow C Stream interface.
    These objects are usually created using `nanoarrow.c_array_stream()`.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> pa_column = pa.array([1, 2, 3], pa.int32())
    >>> pa_batch = pa.record_batch([pa_column], names=["col1"])
    >>> pa_reader = pa.RecordBatchReader.from_batches(pa_batch.schema, [pa_batch])
    >>> array_stream = na.c_array_stream(pa_reader)
    >>> array_stream.get_schema()
    <nanoarrow.c_lib.CSchema struct>
    - format: '+s'
    - name: ''
    - flags: 0
    - metadata: NULL
    - dictionary: NULL
    - children[1]:
      'col1': <nanoarrow.c_lib.CSchema int32>
        - format: 'i'
        - name: 'col1'
        - flags: 2
        - metadata: NULL
        - dictionary: NULL
        - children[0]:
    >>> array_stream.get_next().length
    3
    >>> array_stream.get_next() is None
    Traceback (most recent call last):
      ...
    StopIteration
    """

    if requested_schema is not None:
        requested_schema = c_schema(requested_schema)

    if isinstance(obj, CArrayStream) and requested_schema is None:
        return obj

    if hasattr(obj, "__arrow_c_stream__"):
        requested_schema_capsule = (
            None if requested_schema is None else requested_schema.__arrow_c_schema__()
        )
        return CArrayStream._import_from_c_capsule(
            obj.__arrow_c_stream__(requested_schema=requested_schema_capsule)
        )

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CArrayStream.allocate()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} "
            "to nanoarrow.c_array_stream"
        )


def c_schema_view(obj) -> CSchemaView:
    """ArrowSchemaView wrapper

    The ``ArrowSchemaView`` is a nanoarrow C library structure that facilitates
    access to the deserialized content of an ``ArrowSchema`` (e.g., parameter values for
    parameterized types). This wrapper extends that facility to Python.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.c_schema(pa.decimal128(10, 3))
    >>> schema_view = na.c_schema_view(schema)
    >>> schema_view.type
    'decimal128'
    >>> schema_view.decimal_bitwidth
    128
    >>> schema_view.decimal_precision
    10
    >>> schema_view.decimal_scale
    3
    """

    if isinstance(obj, CSchemaView):
        return obj

    return CSchemaView(c_schema(obj))


def c_array_view(obj, requested_schema=None) -> CArrayView:
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
    >>> array = na.c_array(pa.array(["one", "two", "three", None]))
    >>> array_view = na.c_array_view(array)
    >>> np.array(array_view.buffer(1))
    array([ 0,  3,  6, 11, 11], dtype=int32)
    >>> np.array(array_view.buffer(2))
    array([b'o', b'n', b'e', b't', b'w', b'o', b't', b'h', b'r', b'e', b'e'],
          dtype='|S1')
    """

    if isinstance(obj, CArrayView) and requested_schema is None:
        return obj

    return CArrayView.from_cpu_array(c_array(obj, requested_schema))


def c_buffer(obj, schema=None) -> CBuffer:
    """Owning, read-only ArrowBuffer wrapper

    If obj implement the Python buffer protocol, ``c_buffer()`` Wraps
    obj in nanoarrow's owning buffer structure, the ArrowBuffer,
    such that it can be used to construct arrays. The ownership of the
    underlying buffer is handled by the Python buffer protocol
    (i.e., ``PyObject_GetBuffer()`` and ``PyBuffer_Release()``).

    If obj is iterable, a buffer will be allocated and populated with
    the contents of obj according to ``requested_schema``. The
    ``requested_schema`` parameter is required to create a buffer from
    a Python iterable. The ``struct`` module is currently used to encode
    values from obj into binary form.

    Unlike with :func:`c_array`, ``requested_schema`` is explicitly
    honoured (or an error will be raised).

    Parameters
    ----------

    obj : buffer-like or iterable
        A Python object that supports the Python buffer protocol. This includes
        bytes, memoryview, bytearray, bulit-in types as well as numpy arrays.
    requested_schema :  schema-like, optional
        The data type of the desired buffer as sanitized by
        :func:`c_schema`. Only values that make sense as buffer types are
        allowed (e.g., integer types, floating-point types, interval types,
        decimal types, binary, string, fixed-size binary).

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.c_buffer(b"1234")
    CBuffer(uint8[4 b] 49 50 51 52)
    >>> na.c_buffer([1, 2, 3], na.int32())
    CBuffer(int32[12 b] 1 2 3)
    """
    if isinstance(obj, CBuffer) and requested_schema is None:
        return obj

    if _obj_is_buffer(obj) and requested_schema is None:
        return CBuffer.from_pybuffer(obj)

    if _obj_is_iterable(obj):
        buffer, _ = _c_buffer_from_iterable(obj, requested_schema)
        return buffer

    raise TypeError(
        f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_buffer"
    )


def allocate_c_schema() -> CSchema:
    """Allocate an uninitialized ArrowSchema wrapper

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.allocate_c_schema()
    >>> pa.int32()._export_to_c(schema._addr())
    """
    return CSchema.allocate()


def allocate_c_array(requested_schema=None) -> CArray:
    """Allocate an uninitialized ArrowArray

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.allocate_c_schema()
    >>> pa.int32()._export_to_c(schema._addr())
    """
    if requested_schema is not None:
        requested_schema = c_schema(requested_schema)

    return CArray.allocate(
        CSchema.allocate() if requested_schema is None else requested_schema
    )


def allocate_c_array_stream() -> CArrayStream:
    """Allocate an uninitialized ArrowArrayStream wrapper

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> pa_column = pa.array([1, 2, 3], pa.int32())
    >>> pa_batch = pa.record_batch([pa_column], names=["col1"])
    >>> pa_reader = pa.RecordBatchReader.from_batches(pa_batch.schema, [pa_batch])
    >>> array_stream = na.allocate_c_array_stream()
    >>> pa_reader._export_to_c(array_stream._addr())
    """
    return CArrayStream.allocate()


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


def _obj_is_iterable(obj):
    return hasattr(obj, "__iter__")


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


def _c_array_from_iterable(obj, requested_schema=None):
    if requested_schema is None:
        raise ValueError("requested_schema is required for CArray import from iterable")

    obj_len = -1
    if hasattr(obj, "__len__"):
        obj_len = len(obj)

    # We can always create an array from an empty iterable, even for types
    # not supported by _c_buffer_from_iterable()
    if obj_len == 0:
        builder = CArrayBuilder.allocate()
        builder.init_from_schema(requested_schema)
        builder.start_appending()
        return builder.finish()

    # Use buffer create for crude support of array from iterable
    buffer, n_values = _c_buffer_from_iterable(obj, requested_schema)

    return c_array_from_buffers(
        requested_schema, n_values, buffers=(None, buffer), null_count=0
    )


def _c_buffer_from_iterable(obj, requested_schema=None) -> CBuffer:
    if requested_schema is None:
        raise ValueError("CBuffer from iterable requires requested_schema")

    builder = CBufferBuilder.empty()

    schema_view = c_schema_view(requested_schema)
    if schema_view.storage_type_id != schema_view.type_id:
        raise ValueError(f"Can't create buffer from type {requested_schema}")

    if schema_view.storage_type_id == CArrowType.FIXED_SIZE_BINARY:
        builder.set_data_type(CArrowType.BINARY, schema_view.fixed_size * 8)
    else:
        builder.set_data_type(schema_view.storage_type_id)

    n_values_written = builder.write_values(obj)
    return builder.finish(), n_values_written
