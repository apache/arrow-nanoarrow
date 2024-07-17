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


from nanoarrow._schema import CSchema, CSchemaView
from nanoarrow._utils import obj_is_capsule


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

    if obj_is_capsule(obj, "arrow_schema"):
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


def c_schema_view(obj) -> CSchemaView:
    """ArrowSchemaView wrapper

    The ``ArrowSchemaView`` is a nanoarrow C library structure that facilitates
    access to the deserialized content of an ``ArrowSchema`` (e.g., parameter values for
    parameterized types). This wrapper extends that facility to Python.

    Examples
    --------

    >>> import pyarrow as pa
    >>> import nanoarrow as na
    >>> from nanoarrow.c_schema import c_schema_view
    >>> schema = na.c_schema(pa.decimal128(10, 3))
    >>> schema_view = c_schema_view(schema)
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


def allocate_c_schema() -> CSchema:
    """Allocate an uninitialized ArrowSchema wrapper

    Examples
    --------

    >>> import pyarrow as pa
    >>> from nanoarrow.c_schema import allocate_c_schema
    >>> schema = allocate_c_schema()
    >>> pa.int32()._export_to_c(schema._addr())
    """
    return CSchema.allocate()
