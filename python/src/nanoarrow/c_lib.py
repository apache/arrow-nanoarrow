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

from nanoarrow._lib import CArray, CArrayStream, CArrayView, CSchema, CSchemaView


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

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CSchema.allocate()
        obj._export_to_c(out._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_schema"
        )


def c_array(obj=None, requested_schema=None) -> CArray:
    """ArrowArray wrapper

    This class provides a user-facing interface to access the fields of an ArrowArray
    as defined in the Arrow C Data interface, holding an optional reference to a
    :class:`CSchema` that can be used to safely deserialize the content.

    These objects are created using :func:`c_array`, which accepts any array-like
    object according to the Arrow PyCapsule interface.

    This Python wrapper allows access to array fields but does not automatically
    deserialize their content: use :func:`c_array_view` to validate and deserialize
    the content into a more easily inspectable object.

    Note that the :class:`CArray` objects returned by ``.child()`` hold strong
    references to the original ``ArrowSchema`` to avoid copies while inspecting an
    imported structure.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import numpy as np
    >>> import nanoarrow as na
    >>> array = na.c_array(pa.array(["one", "two", "three", None]))
    >>> array.length
    4
    >>> array.null_count
    1
    """

    if requested_schema is not None:
        requested_schema = c_schema(requested_schema)

    if isinstance(obj, CArray) and requested_schema is None:
        return obj

    if hasattr(obj, "__arrow_c_array__"):
        requested_schema_capsule = (
            None if requested_schema is None else requested_schema.__arrow_c_schema__()
        )
        return CArray._import_from_c_capsule(
            *obj.__arrow_c_array__(requested_schema=requested_schema_capsule)
        )

    # for pyarrow < 14.0
    if hasattr(obj, "_export_to_c"):
        out = CArray.allocate(CSchema.allocate())
        obj._export_to_c(out._addr(), out.schema._addr())
        return out
    else:
        raise TypeError(
            f"Can't convert object of type {type(obj).__name__} to nanoarrow.c_array"
        )


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


def allocate_c_schema():
    """Allocate an uninitialized ArrowSchema wrapper

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> schema = na.allocate_c_schema()
    >>> pa.int32()._export_to_c(schema._addr())
    """
    return CSchema.allocate()


def allocate_c_array(requested_schema=None):
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


def allocate_c_array_stream():
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
