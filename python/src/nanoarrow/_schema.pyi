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

from _typeshed import Incomplete
from typing import ClassVar, Iterable, Mapping

class CArrowTimeUnit:
    """
    Wrapper around ArrowTimeUnit to provide implementations in Python access
    to the values.
    """

    MICRO: ClassVar[int] = ...
    MILLI: ClassVar[int] = ...
    NANO: ClassVar[int] = ...
    SECOND: ClassVar[int] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def __reduce__(self): ...

class CLayout:
    """Abstract buffer information for Arrow types

    Provides accessors for buffer counts, types, and attributes.
    """

    buffer_data_type_id: Incomplete
    child_size_elements: Incomplete
    element_size_bits: Incomplete
    n_buffers: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def __reduce__(self): ...

class CSchema:
    """Low-level ArrowSchema wrapper

    This object is a literal wrapper around a read-only ArrowSchema. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowSchema structures are always released).

    See ``nanoarrow.c_schema()`` for construction and usage examples.
    """

    children: Incomplete
    dictionary: Incomplete
    flags: Incomplete
    format: Incomplete
    metadata: Incomplete
    n_children: Incomplete
    name: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def allocate() -> CSchema:
        """Allocate a released CSchema"""

    def child(self, i): ...
    def is_valid(self) -> bool:
        """Check for a non-null and non-released underlying ArrowSchema"""

    def modify(
        self,
        *,
        format=...,
        name=...,
        flags=...,
        nullable=...,
        metadata=...,
        children=...,
        dictionary=...,
        validate=...
    ) -> CSchema: ...
    def type_equals(self, other, check_nullability: bool = ...) -> bool:
        """Test two schemas for data type equality

        Checks two CSchema objects for type equality (i.e., that an array with
        schema ``actual`` contains elements with the same logical meaning as and
        array with schema ``expected``). Notably, this excludes metadata from
        all nodes in the schema.

        Parameters
        ----------
        other : CSchema
            The schema against which to test
        check_nullability : bool
            If True, actual and expected will be considered equal if their
            data type information and marked nullability are identical.
        """

    def __arrow_c_schema__(self):
        """
        Export to a ArrowSchema PyCapsule
        """

    def __deepcopy__(self, memo=...) -> CSchema: ...
    def __reduce__(self): ...

class CSchemaBuilder:
    """Helper for constructing an ArrowSchema

    The primary function of this class is to wrap the nanoarrow C library calls
    that build up the components of an ArrowSchema.
    """

    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def allocate() -> CSchemaBuilder:
        """Create a CSchemaBuilder

        Allocates memory for an ArrowSchema and populates it with nanoarrow's
        ArrowSchema private_data/release callback implementation. This should
        usually be followed by :meth:`set_type` or :meth:`set_format`.
        """

    def allocate_children(self, n) -> CSchemaBuilder: ...
    def append_metadata(self, metadata: Mapping[bytes, bytes]) -> CSchemaBuilder:
        """Append key/value metadata"""

    def child(self, i) -> CSchemaBuilder: ...
    def finish(self) -> CSchema: ...
    def set_child(self, i, name, child_src) -> CSchemaBuilder: ...
    def set_dictionary(self, dictionary) -> CSchemaBuilder: ...
    def set_dictionary_ordered(self, dictionary_ordered) -> CSchemaBuilder: ...
    def set_flags(self, flags) -> CSchemaBuilder: ...
    def set_format(self, format) -> CSchemaBuilder: ...
    def set_map_keys_sorted(self, map_keys_sorted) -> CSchemaBuilder: ...
    def set_name(self, name) -> CSchemaBuilder: ...
    def set_nullable(self, nullable) -> CSchemaBuilder: ...
    def set_type(self, type_id) -> CSchemaBuilder: ...
    def set_type_date_time(self, type_id, time_unit, timezone) -> CSchemaBuilder: ...
    def set_type_decimal(self, type_id, precision, scale) -> CSchemaBuilder: ...
    def set_type_fixed_size(self, type_id, fixed_size) -> CSchemaBuilder: ...
    def validate(self) -> CSchemaView: ...
    def __reduce__(self): ...

class CSchemaView:
    """Low-level ArrowSchemaView wrapper

    This object is a literal wrapper around a read-only ArrowSchemaView. It provides field accessors
    that return Python objects and handles structure lifecycle. Compared to an ArrowSchema,
    the nanoarrow ArrowSchemaView facilitates access to the deserialized content of an ArrowSchema
    (e.g., parameter values for parameterized types).

    See `nanoarrow.c_schema_view()` for construction and usage examples.
    """

    buffer_format: Incomplete
    decimal_bitwidth: Incomplete
    decimal_precision: Incomplete
    decimal_scale: Incomplete
    dictionary_ordered: Incomplete
    extension_metadata: Incomplete
    extension_name: Incomplete
    fixed_size: Incomplete
    layout: Incomplete
    map_keys_sorted: Incomplete
    nullable: Incomplete
    storage_buffer_format: Incomplete
    storage_type: Incomplete
    storage_type_id: Incomplete
    time_unit: Incomplete
    time_unit_id: Incomplete
    timezone: Incomplete
    type: Incomplete
    type_id: Incomplete
    union_type_ids: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    def __reduce__(self): ...

class SchemaMetadata:
    """Dictionary-like wrapper around a lazily-parsed CSchema.metadata string

    The Arrow C Data interface encodes key/value metadata as a bytes-to-bytes
    mapping using a specific packed binary encoding. This class maintains a
    reference to the underlying storage and parses it as required. Note that
    unlike a Python dictionary, ``SchemaMetadata`` can contain duplicate
    keys.
    """

    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""

    @staticmethod
    def empty():
        """Create an empty SchemaMetadata with no keys or values"""

    def items(self) -> Iterable[tuple[bytes, bytes]]:
        """Iterate over key/value pairs

        The result may contain duplicate keys if they exist in the metadata."""

    def keys(self) -> list[bytes]:
        """List meadata keys

        The result may contain duplicate keys if they exist in the metadata.
        """

    def values(self) -> list[bytes]:
        """List metadata values"""

    def __contains__(self, other) -> bool:
        """Return bool(key in self)."""

    def __getitem__(self, index):
        """Get the value associated with a unique key

        Retrieves the unique value associated with k. Raises KeyError if
        k does not point to exactly one value in the metadata.
        """

    def __iter__(self):
        """Implement iter(self)."""

    def __len__(self) -> int:
        """Return len(self)."""

    def __reduce__(self): ...

def assert_type_equal(actual, expected, check_nullability):
    """Test two schemas for data type equality

    Checks two CSchema objects for type equality (i.e., that an array with
    schema ``actual`` contains elements with the same logical meaning as and
    array with schema ``expected``). Notably, this excludes metadata from
    all nodes in the schema.

    Parameters
    ----------
    actual : CSchema
        The schema to be tested for equality
    expected : CSchema
        The schema against which to test
    check_nullability : bool
        If True, actual and expected will be considered equal if their
        data type information and marked nullability are identical.
    """
