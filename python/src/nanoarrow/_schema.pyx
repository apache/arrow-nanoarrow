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

# cython: language_level = 3

from libc.stdint cimport int32_t, int64_t, uintptr_t
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString, PyBytes_Size
from cpython.pycapsule cimport PyCapsule_GetPointer

from nanoarrow_c cimport (
    ARROW_FLAG_DICTIONARY_ORDERED,
    ARROW_FLAG_MAP_KEYS_SORTED,
    ARROW_FLAG_NULLABLE,
    ArrowFree,
    ArrowLayout,
    ArrowMalloc,
    ArrowMetadataBuilderAppend,
    ArrowMetadataBuilderInit,
    ArrowMetadataReaderInit,
    ArrowMetadataReaderRead,
    ArrowSchema,
    ArrowSchemaAllocateChildren,
    ArrowSchemaAllocateDictionary,
    ArrowSchemaDeepCopy,
    ArrowSchemaInit,
    ArrowSchemaMove,
    ArrowSchemaRelease,
    ArrowSchemaSetMetadata,
    ArrowSchemaSetType,
    ArrowSchemaSetTypeDateTime,
    ArrowSchemaSetTypeDecimal,
    ArrowSchemaSetTypeFixedSize,
    ArrowSchemaSetFormat,
    ArrowSchemaSetName,
    ArrowSchemaToString,
    ArrowSchemaViewInit,
    ArrowStringView,
    ArrowTimeUnit,
    ArrowTimeUnitString,
    ArrowType,
    ArrowTypeString,
    NANOARROW_BUFFER_TYPE_NONE,
    NANOARROW_MAX_FIXED_BUFFERS,
    NANOARROW_TIME_UNIT_SECOND,
    NANOARROW_TIME_UNIT_MILLI,
    NANOARROW_TIME_UNIT_MICRO,
    NANOARROW_TIME_UNIT_NANO,
)

from nanoarrow cimport _types
from nanoarrow._buffer cimport CBuffer
from nanoarrow._utils cimport alloc_c_schema, Error

from typing import Iterable, List, Mapping, Tuple, Union

from nanoarrow import _repr_utils


# This is likely a better fit for a dedicated testing module; however, we need
# it here to produce nice error messages when ensuring that one or
# more arrays conform to a given or inferred schema.
cpdef assert_type_equal(actual, expected, bint check_nullability):
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
    if not isinstance(actual, CSchema):
        raise TypeError(f"actual is {type(actual).__name__}, not CSchema")

    if not isinstance(expected, CSchema):
        raise TypeError(f"expected is {type(expected).__name__}, not CSchema")

    if not actual.type_equals(expected, check_nullability=check_nullability):
        actual_label = actual._to_string(max_chars=80, recursive=True)
        expected_label = expected._to_string(max_chars=80, recursive=True)
        raise ValueError(
            f"Expected schema\n  '{expected_label}'"
            f"\nbut got\n  '{actual_label}'"
        )


cdef class CArrowTimeUnit:
    """
    Wrapper around ArrowTimeUnit to provide implementations in Python access
    to the values.
    """

    SECOND = NANOARROW_TIME_UNIT_SECOND
    MILLI = NANOARROW_TIME_UNIT_MILLI
    MICRO = NANOARROW_TIME_UNIT_MICRO
    NANO = NANOARROW_TIME_UNIT_NANO


cdef class CLayout:
    """Abstract buffer information for Arrow types

    Provides accessors for buffer counts, types, and attributes.
    """

    def __cinit__(self, base, uintptr_t ptr):
        self._base = base
        self._layout = <ArrowLayout*>ptr

        self._n_buffers = NANOARROW_MAX_FIXED_BUFFERS
        for i in range(NANOARROW_MAX_FIXED_BUFFERS):
            if self._layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE:
                self._n_buffers = i
                break

    @property
    def n_buffers(self) -> int:
        return self._n_buffers

    @property
    def buffer_data_type_id(self) -> int:
        return tuple(self._layout.buffer_data_type[i] for i in range(self._n_buffers))

    @property
    def element_size_bits(self) -> int:
        return tuple(self._layout.element_size_bits[i] for i in range(self._n_buffers))

    @property
    def child_size_elements(self) -> int:
        return self._layout.child_size_elements


cdef class SchemaMetadata:
    """Dictionary-like wrapper around a lazily-parsed CSchema.metadata string

    The Arrow C Data interface encodes key/value metadata as a bytes-to-bytes
    mapping using a specific packed binary encoding. This class maintains a
    reference to the underlying storage and parses it as required. Note that
    unlike a Python dictionary, ``SchemaMetadata`` can contain duplicate
    keys.
    """

    def __cinit__(self, object base, uintptr_t ptr):
        self._base = base
        self._metadata = <const char*>ptr

    @staticmethod
    def empty():
        """Create an empty SchemaMetadata with no keys or values"""
        return SchemaMetadata(None, 0)

    cdef _init_reader(self):
        cdef int code = ArrowMetadataReaderInit(&self._reader, self._metadata)
        Error.raise_error_not_ok("ArrowMetadataReaderInit()", code)

    def __len__(self):
        self._init_reader()
        return self._reader.remaining_keys

    def __contains__(self, item):
        for key, _ in self.items():
            if item == key:
                return True

        return False

    def __getitem__(self, k) -> bytes:
        """Get the value associated with a unique key

        Retrieves the unique value associated with k. Raises KeyError if
        k does not point to exactly one value in the metadata.
        """
        out = None

        for key, value in self.items():
            if k == key:
                if out is None:
                    out = value
                else:
                    raise KeyError(f"key {k} matches more than one value in metadata")

        if out is None:
            raise KeyError(f"Key {k} not found")

        return out

    def __iter__(self):
        for key, _ in self.items():
            yield key

    def keys(self) -> List[bytes]:
        """List meadata keys

        The result may contain duplicate keys if they exist in the metadata.
        """
        return list(self)

    def values(self) -> List[bytes]:
        """List metadata values"""
        return [value for _, value in self.items()]

    def items(self) -> Iterable[bytes, bytes]:
        """Iterate over key/value pairs

        The result may contain duplicate keys if they exist in the metadata."""
        cdef ArrowStringView key
        cdef ArrowStringView value
        self._init_reader()
        while self._reader.remaining_keys > 0:
            ArrowMetadataReaderRead(&self._reader, &key, &value)
            key_obj = PyBytes_FromStringAndSize(key.data, key.size_bytes)
            value_obj = PyBytes_FromStringAndSize(value.data, value.size_bytes)
            yield key_obj, value_obj

    def __repr__(self) -> str:
        lines = [
            f"<{_repr_utils.make_class_label(self)}>",
            _repr_utils.metadata_repr(self)
        ]
        return "\n".join(lines)


cdef class CSchema:
    """Low-level ArrowSchema wrapper

    This object is a literal wrapper around a read-only ArrowSchema. It provides field accessors
    that return Python objects and handles the C Data interface lifecycle (i.e., initialized
    ArrowSchema structures are always released).

    See ``nanoarrow.c_schema()`` for construction and usage examples.
    """

    @staticmethod
    def allocate() -> CSchema:
        """Allocate a released CSchema"""
        cdef ArrowSchema* c_schema_out
        base = alloc_c_schema(&c_schema_out)
        return CSchema(base, <uintptr_t>(c_schema_out))

    def __cinit__(self, object base, uintptr_t addr):
        self._base = base
        self._ptr = <ArrowSchema*>addr

    def __deepcopy__(self, memo=None) -> CSchema:
        cdef CSchema out = CSchema.allocate()
        cdef int code = ArrowSchemaDeepCopy(self._ptr, out._ptr)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        return out

    @staticmethod
    def _import_from_c_capsule(schema_capsule) -> CSchema:
        """Import from a ArrowSchema PyCapsule

        Parameters
        ----------
        schema_capsule : PyCapsule
            A valid PyCapsule with name 'arrow_schema' containing an
            ArrowSchema pointer.
        """
        return CSchema(
            schema_capsule,
            <uintptr_t>PyCapsule_GetPointer(schema_capsule, "arrow_schema")
        )

    def __arrow_c_schema__(self):
        """
        Export to a ArrowSchema PyCapsule
        """
        self._assert_valid()

        cdef ArrowSchema* c_schema_out
        schema_capsule = alloc_c_schema(&c_schema_out)

        cdef int code = ArrowSchemaDeepCopy(self._ptr, c_schema_out)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy", code)
        return schema_capsule

    @property
    def _capsule(self):
        """
        Returns the capsule backing this CSchema or None if it does not exist
        or points to a parent ArrowSchema.
        """
        cdef ArrowSchema* maybe_capsule_ptr
        maybe_capsule_ptr = <ArrowSchema*>PyCapsule_GetPointer(self._base, 'arrow_schema')

        # This will return False if this is a child CSchema whose capsule holds
        # the parent ArrowSchema
        if maybe_capsule_ptr == self._ptr:
            return self._base

        return None

    def _addr(self) -> int:
        return <uintptr_t>self._ptr

    def is_valid(self) -> bool:
        """Check for a non-null and non-released underlying ArrowSchema"""
        return self._ptr != NULL and self._ptr.release != NULL

    def _assert_valid(self):
        if self._ptr == NULL:
            raise RuntimeError("schema is NULL")
        if self._ptr.release == NULL:
            raise RuntimeError("schema is released")

    def _to_string(self, int64_t max_chars=0, recursive=False) -> str:
        cdef int64_t n_chars
        if max_chars == 0:
            n_chars = ArrowSchemaToString(self._ptr, NULL, 0, recursive)
        else:
            n_chars = max_chars

        cdef char* out = <char*>ArrowMalloc(n_chars + 1)
        if not out:
            raise MemoryError()

        ArrowSchemaToString(self._ptr, out, n_chars + 1, recursive)
        out_str = out.decode("UTF-8")
        ArrowFree(out)

        return out_str

    def __repr__(self) -> str:
        return _repr_utils.schema_repr(self)

    def type_equals(self, CSchema other, check_nullability: bool=False) -> bool:
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
        self._assert_valid()

        if self._ptr == other._ptr:
            return True

        if self.format != other.format:
            return False

        # Nullability is not strictly part of the "type"; however, performing
        # this check recursively is verbose to otherwise accomplish and
        # sometimes this does matter.
        cdef int64_t flags = self.flags
        cdef int64_t other_flags = other.flags
        if not check_nullability:
            flags &= ~ARROW_FLAG_NULLABLE
            other_flags &= ~ARROW_FLAG_NULLABLE

        if flags != other_flags:
            return False

        if self.n_children != other.n_children:
            return False

        for child, other_child in zip(self.children, other.children):
            if not child.type_equals(other_child, check_nullability=check_nullability):
                return False

        if (self.dictionary is None) != (other.dictionary is None):
            return False

        if self.dictionary is not None:
            if not self.dictionary.type_equals(
                other.dictionary,
                check_nullability=check_nullability
            ):
                return False

        return True


    @property
    def format(self) -> str:
        self._assert_valid()
        if self._ptr.format != NULL:
            return self._ptr.format.decode()

    @property
    def name(self) -> Union[str, None]:
        self._assert_valid()
        if self._ptr.name != NULL:
            return self._ptr.name.decode()
        else:
            return None

    @property
    def flags(self) -> int:
        return self._ptr.flags

    @property
    def metadata(self) -> SchemaMetadata:
        self._assert_valid()
        if self._ptr.metadata != NULL:
            return SchemaMetadata(self._base, <uintptr_t>self._ptr.metadata)
        else:
            return None

    @property
    def n_children(self) -> int:
        self._assert_valid()
        return self._ptr.n_children

    def child(self, int64_t i):
        self._assert_valid()
        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"{i} out of range [0, {self._ptr.n_children})")

        return CSchema(self._base, <uintptr_t>self._ptr.children[i])

    @property
    def children(self) -> Iterable[CSchema]:
        for i in range(self.n_children):
            yield self.child(i)

    @property
    def dictionary(self) -> Union[CSchema, None]:
        self._assert_valid()
        if self._ptr.dictionary != NULL:
            return CSchema(self, <uintptr_t>self._ptr.dictionary)
        else:
            return None

    def modify(self, *, format=None, name=None, flags=None, nullable=None,
               metadata=None, children=None, dictionary=None, validate=True) -> CSchema:
        cdef CSchemaBuilder builder = CSchemaBuilder.allocate()

        if format is None:
            builder.set_format(self.format)
        else:
            builder.set_format(format)

        if name is None:
            builder.set_name(self.name)
        elif name is not False:
            builder.set_name(name)

        if flags is None:
            builder.set_flags(self.flags)
        else:
            builder.set_flags(flags)

        if nullable is not None:
            builder.set_nullable(nullable)

        if metadata is None:
            if self.metadata is not None:
                builder.append_metadata(self.metadata)
        else:
            builder.append_metadata(metadata)

        if children is None:
            if self.n_children > 0:
                builder.allocate_children(self.n_children)
                for i, child in enumerate(self.children):
                    builder.set_child(i, None, child)
        elif hasattr(children, "items"):
            builder.allocate_children(len(children))
            for i, item in enumerate(children.items()):
                name, child = item
                builder.set_child(i, name, child)
        else:
            builder.allocate_children(len(children))
            for i, child in enumerate(children):
                builder.set_child(i, None, child)

        if dictionary is None:
            if self.dictionary:
                builder.set_dictionary(self.dictionary)
        elif dictionary is not False:
            builder.set_dictionary(dictionary)

        if validate:
            builder.validate()

        return builder.finish()


cdef class CSchemaView:
    """Low-level ArrowSchemaView wrapper

    This object is a literal wrapper around a read-only ArrowSchemaView. It provides field accessors
    that return Python objects and handles structure lifecycle. Compared to an ArrowSchema,
    the nanoarrow ArrowSchemaView facilitates access to the deserialized content of an ArrowSchema
    (e.g., parameter values for parameterized types).

    See `nanoarrow.c_schema_view()` for construction and usage examples.
    """

    def __cinit__(self, CSchema schema):
        self._base = schema
        self._schema_view.type = <ArrowType>_types.UNINITIALIZED
        self._schema_view.storage_type = <ArrowType>_types.UNINITIALIZED

        cdef Error error = Error()
        cdef int code = ArrowSchemaViewInit(&self._schema_view, schema._ptr, &error.c_error)
        error.raise_message_not_ok("ArrowSchemaViewInit()", code)

        self._dictionary_ordered = schema._ptr.flags & ARROW_FLAG_DICTIONARY_ORDERED
        self._nullable = schema._ptr.flags & ARROW_FLAG_NULLABLE
        self._map_keys_sorted = schema._ptr.flags & ARROW_FLAG_MAP_KEYS_SORTED

    @property
    def layout(self) -> CLayout:
        return CLayout(self, <uintptr_t>&self._schema_view.layout)

    @property
    def type_id(self) -> int:
        return self._schema_view.type

    @property
    def storage_type_id(self) -> int:
        return self._schema_view.storage_type

    @property
    def storage_buffer_format(self) -> Union[str, None]:
        if self.buffer_format is not None:
            return self.buffer_format
        elif _types.equal(self._schema_view.type, _types.DATE32):
            return 'i'
        elif _types.one_of(
            self._schema_view.type,
            (_types.TIMESTAMP, _types.DATE64, _types.DURATION)
        ):
            return 'q'
        else:
            return None

    @property
    def buffer_format(self) -> Union[str, None]:
        """The Python struct format representing an element of this type
        or None if there is no Python format string that can represent this
        type without loosing information.
        """
        if self.extension_name or self._schema_view.type != self._schema_view.storage_type:
            return None

        # String/binary types do not have format strings as far as the Python
        # buffer protocol is concerned
        if self.layout.n_buffers != 2:
            return None

        cdef char out[128]
        cdef int element_size_bits = 0
        if _types.equal(self._schema_view.type, _types.FIXED_SIZE_BINARY):
            element_size_bits = self._schema_view.fixed_size * 8

        try:
            _types.to_format(self._schema_view.type, element_size_bits, sizeof(out), out)
            return out.decode()
        except ValueError:
            return None

    @property
    def type(self) -> str:
        cdef const char* type_str = ArrowTypeString(self._schema_view.type)
        if type_str != NULL:
            return type_str.decode()
        else:
            raise ValueError("ArrowTypeString() returned NULL")

    @property
    def storage_type(self) -> str:
        cdef const char* type_str = ArrowTypeString(self._schema_view.storage_type)
        if type_str != NULL:
            return type_str.decode()
        else:
            raise ValueError("ArrowTypeString() returned NULL")

    @property
    def dictionary_ordered(self) -> Union[bool, None]:
        if _types.equal(self._schema_view.type, _types.DICTIONARY):
            return self._dictionary_ordered != 0
        else:
            return None

    @property
    def nullable(self) -> bool:
        return self._nullable != 0

    @property
    def map_keys_sorted(self) -> Union[bool, None]:
        if _types.equal(self._schema_view.type, _types.MAP):
            return self._map_keys_sorted != 0
        else:
            return None

    @property
    def fixed_size(self) -> Union[bool, None]:
        if _types.is_fixed_size(self._schema_view.type):
            return self._schema_view.fixed_size
        else:
            return None

    @property
    def decimal_bitwidth(self) -> Union[int, None]:
        if _types.is_decimal(self._schema_view.type):
            return self._schema_view.decimal_bitwidth
        else:
            return None

    @property
    def decimal_precision(self) -> Union[int, None]:
        if _types.is_decimal(self._schema_view.type):
            return self._schema_view.decimal_precision
        else:
            return None

    @property
    def decimal_scale(self) -> Union[int, None]:
        if _types.is_decimal(self._schema_view.type):
            return self._schema_view.decimal_scale
        else:
            return None

    @property
    def time_unit_id(self) -> Union[int, None]:
        if _types.has_time_unit(self._schema_view.type):
            return self._schema_view.time_unit
        else:
            return None

    @property
    def time_unit(self) -> Union[str, None]:
        if _types.has_time_unit(self._schema_view.type):
            return ArrowTimeUnitString(self._schema_view.time_unit).decode()
        else:
            return None

    @property
    def timezone(self) -> Union[str, None]:
        if _types.equal(self._schema_view.type, _types.TIMESTAMP):
            return self._schema_view.timezone.decode()
        else:
            return None

    @property
    def union_type_ids(self) -> Union[Tuple[int, ...], None]:
        if _types.is_union(self._schema_view.type):
            type_ids_str = self._schema_view.union_type_ids.decode().split(',')
            return (int(type_id) for type_id in type_ids_str)
        else:
            return None

    @property
    def extension_name(self) -> Union[str, None]:
        if self._schema_view.extension_name.data != NULL:
            name_bytes = PyBytes_FromStringAndSize(
                self._schema_view.extension_name.data,
                self._schema_view.extension_name.size_bytes
            )
            return name_bytes.decode()
        else:
            return None

    @property
    def extension_metadata(self) -> Union[bytes, None]:
        if self._schema_view.extension_name.data != NULL:
            return PyBytes_FromStringAndSize(
                self._schema_view.extension_metadata.data,
                self._schema_view.extension_metadata.size_bytes
            )
        else:
            return None

    def __repr__(self) -> str:
        return _repr_utils.schema_view_repr(self)


cdef class CSchemaBuilder:
    """Helper for constructing an ArrowSchema

    The primary function of this class is to wrap the nanoarrow C library calls
    that build up the components of an ArrowSchema.
    """

    def __cinit__(self, CSchema schema):
        self.c_schema = schema
        self._ptr = schema._ptr
        if self._ptr.release == NULL:
            ArrowSchemaInit(self._ptr)

    @staticmethod
    def allocate() -> CSchemaBuilder:
        """Create a CSchemaBuilder

        Allocates memory for an ArrowSchema and populates it with nanoarrow's
        ArrowSchema private_data/release callback implementation. This should
        usually be followed by :meth:`set_type` or :meth:`set_format`.
        """
        return CSchemaBuilder(CSchema.allocate())

    def append_metadata(self, metadata: Mapping[bytes, bytes]) -> CSchemaBuilder:
        """Append key/value metadata"""
        cdef CBuffer buffer = CBuffer.empty()

        cdef const char* existing_metadata = self.c_schema._ptr.metadata
        cdef int code = ArrowMetadataBuilderInit(buffer._ptr, existing_metadata)
        Error.raise_error_not_ok("ArrowMetadataBuilderInit()", code)

        cdef ArrowStringView key
        cdef ArrowStringView value
        cdef int32_t keys_added = 0

        for k, v in metadata.items():
            k = k.encode() if isinstance(k, str) else bytes(k)
            key.data = PyBytes_AsString(k)
            key.size_bytes = PyBytes_Size(k)

            v = v.encode() if isinstance(v, str) else bytes(v)
            value.data = PyBytes_AsString(v)
            value.size_bytes = PyBytes_Size(v)

            code = ArrowMetadataBuilderAppend(buffer._ptr, key, value)
            Error.raise_error_not_ok("ArrowMetadataBuilderAppend()", code)

            keys_added += 1

        if keys_added > 0:
            code = ArrowSchemaSetMetadata(self.c_schema._ptr, <const char*>buffer._ptr.data)
            Error.raise_error_not_ok("ArrowSchemaSetMetadata()", code)

        return self

    def child(self, int64_t i) -> CSchemaBuilder:
        return CSchemaBuilder(self.c_schema.child(i))

    def set_type(self, int type_id) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetType(self._ptr, <ArrowType>type_id)
        Error.raise_error_not_ok("ArrowSchemaSetType()", code)

        return self

    def set_type_decimal(self, int type_id, int precision, int scale) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetTypeDecimal(self._ptr, <ArrowType>type_id, precision, scale)
        Error.raise_error_not_ok("ArrowSchemaSetType()", code)

    def set_type_fixed_size(self, int type_id, int fixed_size) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetTypeFixedSize(self._ptr, <ArrowType>type_id, fixed_size)
        Error.raise_error_not_ok("ArrowSchemaSetTypeFixedSize()", code)

        return self

    def set_type_date_time(self, int type_id, int time_unit, timezone) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code
        if timezone is None:
            code = ArrowSchemaSetTypeDateTime(self._ptr, <ArrowType>type_id, <ArrowTimeUnit>time_unit, NULL)
        else:
            timezone = str(timezone)
            code = ArrowSchemaSetTypeDateTime(self._ptr, <ArrowType>type_id, <ArrowTimeUnit>time_unit, timezone.encode("UTF-8"))

        Error.raise_error_not_ok("ArrowSchemaSetTypeDateTime()", code)

        return self

    def set_format(self, str format) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaSetFormat(self._ptr, format.encode("UTF-8"))
        Error.raise_error_not_ok("ArrowSchemaSetFormat()", code)

        return self

    def set_name(self, name) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code
        if name is None:
            code = ArrowSchemaSetName(self._ptr, NULL)
        else:
            name = str(name)
            code = ArrowSchemaSetName(self._ptr, name.encode("UTF-8"))

        Error.raise_error_not_ok("ArrowSchemaSetName()", code)

        return self

    def allocate_children(self, int n) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code = ArrowSchemaAllocateChildren(self._ptr, n)
        Error.raise_error_not_ok("ArrowSchemaAllocateChildren()", code)

        return self

    def set_child(self, int64_t i, name, CSchema child_src) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        if i < 0 or i >= self._ptr.n_children:
            raise IndexError(f"Index out of range: {i}")

        if self._ptr.children[i].release != NULL:
            ArrowSchemaRelease(self._ptr.children[i])

        cdef int code = ArrowSchemaDeepCopy(child_src._ptr, self._ptr.children[i])
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        if name is not None:
            name = str(name)
            code = ArrowSchemaSetName(self._ptr.children[i], name.encode("UTF-8"))
            Error.raise_error_not_ok("ArrowSchemaSetName()", code)

        return self

    def set_dictionary(self, CSchema dictionary) -> CSchemaBuilder:
        self.c_schema._assert_valid()

        cdef int code
        if self._ptr.dictionary == NULL:
            code = ArrowSchemaAllocateDictionary(self._ptr)
            Error.raise_error_not_ok("ArrowSchemaAllocateDictionary()", code)

        if self._ptr.dictionary.release != NULL:
            ArrowSchemaRelease(self._ptr.dictionary)

        code = ArrowSchemaDeepCopy(dictionary._ptr, self._ptr.dictionary)
        Error.raise_error_not_ok("ArrowSchemaDeepCopy()", code)

        return self

    def set_flags(self, flags) -> CSchemaBuilder:
        self._ptr.flags = flags
        return self

    def set_nullable(self, nullable) -> CSchemaBuilder:
        if nullable:
            self._ptr.flags = self._ptr.flags | ARROW_FLAG_NULLABLE
        else:
            self._ptr.flags = self._ptr.flags & ~ARROW_FLAG_NULLABLE

        return self

    def set_dictionary_ordered(self, dictionary_ordered) -> CSchemaBuilder:
        if dictionary_ordered:
            self._ptr.flags = self._ptr.flags | ARROW_FLAG_DICTIONARY_ORDERED
        else:
            self._ptr.flags = self._ptr.flags & ~ARROW_FLAG_DICTIONARY_ORDERED

        return self

    def validate(self) -> CSchemaView:
        return CSchemaView(self.c_schema)

    def finish(self) -> CSchema:
        self.c_schema._assert_valid()
        cdef CSchema out = CSchema.allocate()
        ArrowSchemaMove(self.c_schema._ptr, out._ptr)
        ArrowSchemaInit(self.c_schema._ptr)
        return out
