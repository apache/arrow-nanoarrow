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

from nanoarrow._array_stream import CArrayStream
from nanoarrow._utils import obj_is_capsule
from nanoarrow.c_array import c_array
from nanoarrow.c_schema import c_schema


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
    <nanoarrow.c_schema.CSchema struct>
    - format: '+s'
    - name: ''
    - flags: 0
    - metadata: NULL
    - dictionary: NULL
    - children[1]:
      'col1': <nanoarrow.c_schema.CSchema int32>
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
    if obj_is_capsule(obj, "arrow_array_stream"):
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
        return CArrayStream.from_c_arrays([array], array.schema, validate=False)
    except Exception as e:
        raise TypeError(
            f"An error occurred whilst converting {type(obj).__name__} "
            f"to nanoarrow.c_array_stream or nanoarrow.c_array: \n {e}"
        ) from e


def allocate_c_array_stream() -> CArrayStream:
    """Allocate an uninitialized ArrowArrayStream wrapper

    Examples
    --------

    >>> import pyarrow as pa
    >>> from nanoarrow.c_array_stream import allocate_c_array_stream
    >>> pa_column = pa.array([1, 2, 3], pa.int32())
    >>> pa_batch = pa.record_batch([pa_column], names=["col1"])
    >>> pa_reader = pa.RecordBatchReader.from_batches(pa_batch.schema, [pa_batch])
    >>> array_stream = allocate_c_array_stream()
    >>> pa_reader._export_to_c(array_stream._addr())
    """
    return CArrayStream.allocate()


def _obj_is_pyarrow_record_batch_reader(obj):
    obj_type = type(obj)
    if not obj_type.__module__.startswith("pyarrow"):
        return False

    if not obj_type.__name__.endswith("RecordBatchReader"):
        return False

    return hasattr(obj, "_export_to_c")
