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

import _cython_3_0_11
from _typeshed import Incomplete
from typing import ClassVar

__pyx_capi__: dict
__reduce_cython__: _cython_3_0_11.cython_function_or_method
__setstate_cython__: _cython_3_0_11.cython_function_or_method
__test__: dict
assert_type_equal: _cython_3_0_11.cython_function_or_method

class CArrowTimeUnit:
    MICRO: ClassVar[int] = ...
    MILLI: ClassVar[int] = ...
    NANO: ClassVar[int] = ...
    SECOND: ClassVar[int] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...
    def __reduce_cython__(self, *args, **kwargs): ...
    def __setstate_cython__(self, *args, **kwargs): ...

class CLayout:
    buffer_data_type_id: Incomplete
    child_size_elements: Incomplete
    element_size_bits: Incomplete
    n_buffers: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    def __reduce__(self): ...

class CSchema:
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
    def allocate(*args, **kwargs):
        """Allocate a released CSchema"""
    def child(self, *args, **kwargs): ...
    def is_valid(self, *args, **kwargs):
        """Check for a non-null and non-released underlying ArrowSchema"""
    def modify(self, *args, **kwargs): ...
    def type_equals(self, *args, **kwargs):
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
    def __arrow_c_schema__(self, *args, **kwargs):
        """
        Export to a ArrowSchema PyCapsule
        """
    def __deepcopy__(self): ...
    def __reduce__(self): ...

class CSchemaBuilder:
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def allocate(*args, **kwargs):
        """Create a CSchemaBuilder

        Allocates memory for an ArrowSchema and populates it with nanoarrow's
        ArrowSchema private_data/release callback implementation. This should
        usually be followed by :meth:`set_type` or :meth:`set_format`.
        """
    def allocate_children(self, *args, **kwargs): ...
    def append_metadata(self, *args, **kwargs):
        """Append key/value metadata"""
    def child(self, *args, **kwargs): ...
    def finish(self, *args, **kwargs): ...
    def set_child(self, *args, **kwargs): ...
    def set_dictionary(self, *args, **kwargs): ...
    def set_dictionary_ordered(self, *args, **kwargs): ...
    def set_flags(self, *args, **kwargs): ...
    def set_format(self, *args, **kwargs): ...
    def set_name(self, *args, **kwargs): ...
    def set_nullable(self, *args, **kwargs): ...
    def set_type(self, *args, **kwargs): ...
    def set_type_date_time(self, *args, **kwargs): ...
    def set_type_decimal(self, *args, **kwargs): ...
    def set_type_fixed_size(self, *args, **kwargs): ...
    def validate(self, *args, **kwargs): ...
    def __reduce__(self): ...

class CSchemaView:
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
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
    @staticmethod
    def empty(*args, **kwargs):
        """Create an empty SchemaMetadata with no keys or values"""
    def items(self, *args, **kwargs):
        """Iterate over key/value pairs

        The result may contain duplicate keys if they exist in the metadata."""
    def keys(self, *args, **kwargs):
        """List meadata keys

        The result may contain duplicate keys if they exist in the metadata.
        """
    def values(self, *args, **kwargs):
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
