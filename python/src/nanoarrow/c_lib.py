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


from nanoarrow._lib import (
    CArray,
    CArrayStream,
    CArrayView,
    CBuffer,
    CSchema,
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
    if _obj_is_capsule(obj, "arrow_array"):
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

    # e.g., iterable, empty
    return _get_builder().build_c_array(obj, schema)


def c_array_stream(obj=None, schema=None) -> CArrayStream:
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

    if schema is not None:
        schema = c_schema(schema)

    if isinstance(obj, CArrayStream) and schema is None:
        return obj

    # Try capsule protocol
    if hasattr(obj, "__arrow_c_stream__"):
        schema_capsule = None if schema is None else schema.__arrow_c_schema__()
        return CArrayStream._import_from_c_capsule(
            obj.__arrow_c_stream__(requested_schema=schema_capsule)
        )

    # Try import of bare capsule
    if _obj_is_capsule(obj, "arrow_array_stream"):
        if schema is not None:
            raise TypeError(
                "Can't import c_array_stream from capsule with requested schema"
            )
        return CArrayStream._import_from_c_capsule(obj)

    # Try _export_to_c for RecordBatchReader objects if pyarrow < 14.0
    if _obj_is_pyarrow_record_batch_reader(obj):
        out = CArrayStream.allocate()
        obj._export_to_c(out._addr())
        return out

    try:
        array = c_array(obj, schema=schema)
        return CArrayStream.from_array_list([array], array.schema, validate=False)
    except Exception as e:
        raise TypeError(
            f"An error occurred whilst converting {type(obj).__name__} "
            f"to nanoarrow.c_array_stream or nanoarrow.c_array: \n {e}"
        ) from e


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
    >>> array = na.c_array(pa.array(["one", "two", "three", None]))
    >>> array_view = na.c_array_view(array)
    >>> np.array(array_view.buffer(1))
    array([ 0,  3,  6, 11, 11], dtype=int32)
    >>> np.array(array_view.buffer(2))
    array([b'o', b'n', b'e', b't', b'w', b'o', b't', b'h', b'r', b'e', b'e'],
          dtype='|S1')
    """

    if isinstance(obj, CArrayView) and schema is None:
        return obj

    return c_array(obj, schema).view()


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

    obj : buffer-like or iterable
        A Python object that supports the Python buffer protocol. This includes
        bytes, memoryview, bytearray, bulit-in types as well as numpy arrays.
    schema :  schema-like, optional
        The data type of the desired buffer as sanitized by
        :func:`c_schema`. Only values that make sense as buffer types are
        allowed (e.g., integer types, floating-point types, interval types,
        decimal types, binary, string, fixed-size binary).

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.c_buffer(b"1234")
    nanoarrow.c_lib.CBuffer(uint8[4 b] 49 50 51 52)
    >>> na.c_buffer([1, 2, 3], na.int32())
    nanoarrow.c_lib.CBuffer(int32[12 b] 1 2 3)
    """
    if isinstance(obj, CBuffer) and schema is None:
        return obj

    if _obj_is_buffer(obj):
        if schema is not None:
            raise NotImplementedError(
                "c_buffer() with schema for pybuffer is not implemented"
            )
        return CBuffer.from_pybuffer(obj)

    if _obj_is_iterable(obj):
        buffer, _ = _c_buffer_from_iterable(obj, schema)
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


def allocate_c_array(schema=None) -> CArray:
    """Allocate an uninitialized ArrowArray

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.allocate_c_schema()
    >>> pa.int32()._export_to_c(schema._addr())
    """
    if schema is not None:
        schema = c_schema(schema)

    return CArray.allocate(CSchema.allocate() if schema is None else schema)


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


def _obj_is_pyarrow_record_batch_reader(obj):
    obj_type = type(obj)
    if not obj_type.__module__.startswith("pyarrow"):
        return False

    if not obj_type.__name__.endswith("RecordBatchReader"):
        return False

    return hasattr(obj, "_export_to_c")


def _obj_is_iterable(obj):
    return hasattr(obj, "__iter__")


# To prevent a cyclical import with the builder module, we define
# wrappers here.
_builder = None


def _get_builder():
    global _builder
    if _builder is None:
        from nanoarrow import builder as _builder
    return _builder


def _c_buffer_from_iterable(obj, schema):
    return _get_builder()._c_buffer_from_iterable(obj, schema)
