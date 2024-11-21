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

import enum
import reprlib
from functools import cached_property
from typing import List, Mapping, Union

from nanoarrow._schema import (
    CArrowTimeUnit,
    CSchemaBuilder,
    CSchemaView,
    SchemaMetadata,
)
from nanoarrow.c_schema import c_schema
from nanoarrow.extension import resolve_extension

from nanoarrow import _repr_utils, _types


class Type(enum.Enum):
    """The Type enumerator provides a means by which the various type
    categories can be identified. Type values can be used in place of
    :class:`Schema` instances in most places for parameter-free types.
    """

    UNINITIALIZED = int(_types.UNINITIALIZED)
    NULL = int(_types.NA)
    BOOL = int(_types.BOOL)
    UINT8 = int(_types.UINT8)
    INT8 = int(_types.INT8)
    UINT16 = int(_types.UINT16)
    INT16 = int(_types.INT16)
    UINT32 = int(_types.UINT32)
    INT32 = int(_types.INT32)
    UINT64 = int(_types.UINT64)
    INT64 = int(_types.INT64)
    HALF_FLOAT = int(_types.HALF_FLOAT)
    FLOAT = int(_types.FLOAT)
    DOUBLE = int(_types.DOUBLE)
    STRING = int(_types.STRING)
    BINARY = int(_types.BINARY)
    FIXED_SIZE_BINARY = int(_types.FIXED_SIZE_BINARY)
    DATE32 = int(_types.DATE32)
    DATE64 = int(_types.DATE64)
    TIMESTAMP = int(_types.TIMESTAMP)
    TIME32 = int(_types.TIME32)
    TIME64 = int(_types.TIME64)
    INTERVAL_MONTHS = int(_types.INTERVAL_MONTHS)
    INTERVAL_DAY_TIME = int(_types.INTERVAL_DAY_TIME)
    DECIMAL128 = int(_types.DECIMAL128)
    DECIMAL256 = int(_types.DECIMAL256)
    LIST = int(_types.LIST)
    STRUCT = int(_types.STRUCT)
    SPARSE_UNION = int(_types.SPARSE_UNION)
    DENSE_UNION = int(_types.DENSE_UNION)
    DICTIONARY = int(_types.DICTIONARY)
    MAP = int(_types.MAP)
    EXTENSION = int(_types.EXTENSION)
    FIXED_SIZE_LIST = int(_types.FIXED_SIZE_LIST)
    DURATION = int(_types.DURATION)
    LARGE_STRING = int(_types.LARGE_STRING)
    LARGE_BINARY = int(_types.LARGE_BINARY)
    LARGE_LIST = int(_types.LARGE_LIST)
    INTERVAL_MONTH_DAY_NANO = int(_types.INTERVAL_MONTH_DAY_NANO)
    RUN_END_ENCODED = int(_types.RUN_END_ENCODED)
    BINARY_VIEW = int(_types.BINARY_VIEW)
    STRING_VIEW = int(_types.STRING_VIEW)

    def __arrow_c_schema__(self):
        # This will only work for parameter-free types
        c_schema = CSchemaBuilder.allocate().set_type(self.value).set_name("").finish()
        return c_schema._capsule


class TimeUnit(enum.Enum):
    """Unit enumerator for timestamp, duration, and time types."""

    SECOND = CArrowTimeUnit.SECOND
    MILLI = CArrowTimeUnit.MILLI
    MICRO = CArrowTimeUnit.MICRO
    NANO = CArrowTimeUnit.NANO

    @staticmethod
    def create(obj):
        """Create a TimeUnit from parameter input.

        This constructor will accept the abbreviations "s", "ms", "us", and "ns"
        and return the appropriate enumerator value.

        >>> import nanoarrow as na
        >>> na.TimeUnit.create("s")
        <TimeUnit.SECOND: 0>
        """

        if isinstance(obj, str):
            if obj == "s":
                return TimeUnit.SECOND
            elif obj == "ms":
                return TimeUnit.MILLI
            elif obj == "us":
                return TimeUnit.MICRO
            elif obj == "ns":
                return TimeUnit.NANO

        return TimeUnit(obj)


class ExtensionAccessor:
    """Accessor for extension type parameters"""

    def __init__(self, schema) -> None:
        self._schema = schema
        self._ext = resolve_extension(self._schema._c_schema_view)
        self._params = self._ext.get_params(self._schema) if self._ext else {}

    def __dir__(self) -> List[str]:
        return ["name", "metadata", "storage"] + list(self._params.keys())

    @property
    def name(self) -> str:
        """Extension name for this extension type"""
        return self._schema._c_schema_view.extension_name

    @property
    def metadata(self) -> bytes:
        """Extension metadata for this extension type if present"""
        extension_metadata = self._schema._c_schema_view.extension_metadata
        return extension_metadata if extension_metadata else b""

    @property
    def storage(self):
        """Storage type for this extension type"""
        metadata = dict(self._schema.metadata.items())

        # Remove metadata keys that cause this type to be treated as an extension
        del metadata[b"ARROW:extension:name"]
        if b"ARROW:extension:metadata" in metadata:
            del metadata[b"ARROW:extension:metadata"]

        return Schema(self._schema, metadata=metadata)

    def __getattr__(self, key: str):
        return self._params[key]


class Schema:
    """Create a nanoarrow Schema

    The Schema is nanoarrow's high-level data type representation, encompassing
    the role of PyArrow's ``Schema``, ``Field``, and ``DataType``. This scope
    maps to that of the ArrowSchema in the Arrow C Data interface.

    Parameters
    ----------
    obj :
        A :class:`Type` specifier or a schema-like object. A schema-like object
        includes:
        * A ``pyarrow.Schema``, `pyarrow.Field``, or ``pyarrow.DataType``
        * A nanoarrow :class:`Schema`, :class:`CSchema`, or :class:`Type`
        * Any object implementing the Arrow PyCapsule interface protocol method.

    name : str, optional
        An optional name to bind to this field.

    nullable : bool, optional
        Explicitly specify field nullability. Fields are nullable by default.

    metadata : mapping, optional
        Explicitly specify field metadata.

    params :
        Type-specific parameters when ``obj`` is a :class:`Type`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> import pyarrow as pa
    >>> na.Schema(na.Type.INT32)
    <Schema> int32
    >>> na.Schema(na.Type.DURATION, unit=na.TimeUnit.SECOND)
    <Schema> duration('s')
    >>> na.Schema(pa.int32())
    <Schema> int32
    """

    def __init__(
        self,
        obj,
        *,
        name=None,
        nullable=None,
        metadata=None,
        fields=None,
        **params,
    ) -> None:
        if isinstance(obj, Type):
            self._c_schema = _c_schema_from_type_and_params(obj, params)
        else:
            if params:
                raise ValueError("params are only supported for obj of class Type")
            self._c_schema = c_schema(obj)

        if (
            name is not None
            or nullable is not None
            or metadata is not None
            or fields is not None
        ):
            self._c_schema = self._c_schema.modify(
                name=name,
                nullable=nullable,
                metadata=metadata,
                children=_clean_fields(fields),
            )

        self._c_schema_view = CSchemaView(self._c_schema)

    @property
    def params(self) -> Mapping:
        """Get parameter names and values for this type

        Returns a dictionary of parameters that can be used to reconstruct
        this type together with its type identifier.

        >>> import nanoarrow as na
        >>> na.fixed_size_binary(123).params
        {'byte_width': 123}
        """
        if self._c_schema_view.type_id not in _PARAM_NAMES:
            return {}

        param_names = _PARAM_NAMES[self._c_schema_view.type_id]
        return {k: getattr(self, k) for k in param_names}

    @property
    def type(self) -> Type:
        """Type enumerator value of this Schema

        >>> import nanoarrow as na
        >>> na.int32().type
        <Type.INT32: 8>
        """
        if self._c_schema_view.extension_name:
            return Type.EXTENSION
        else:
            return Type(self._c_schema_view.type_id)

    @property
    def name(self) -> Union[str, None]:
        """Field name of this Schema

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.field(0).name
        'col1'
        """
        return self._c_schema.name

    @property
    def nullable(self) -> bool:
        """Nullability of this field

        >>> import nanoarrow as na
        >>> na.int32().nullable
        True
        >>> na.int32(nullable=False).nullable
        False
        """
        return self._c_schema_view.nullable

    @cached_property
    def metadata(self) -> Mapping[bytes, bytes]:
        """Access field metadata of this field

        >>> import nanoarrow as na
        >>> schema = na.Schema(na.int32(), metadata={"key": "value"})
        >>> dict(schema.metadata.items())
        {b'key': b'value'}
        """
        c_schema_metadata = self._c_schema.metadata
        return (
            SchemaMetadata.empty() if c_schema_metadata is None else c_schema_metadata
        )

    @cached_property
    def extension(self) -> Union[ExtensionAccessor, None]:
        """Access extension type attributes

        >>> import nanoarrow as na
        >>> schema = na.extension_type(na.int32(), "arrow.example", b"{}")
        >>> schema.extension.name
        'arrow.example'
        >>> schema.extension.metadata
        b'{}'
        """
        extension_name = self._c_schema_view.extension_name
        if extension_name:
            return ExtensionAccessor(self)

    @property
    def byte_width(self) -> Union[int, None]:
        """Element byte width for fixed-size binary type

        Returns ``None`` for types for which this property is not relevant.

        >>> import nanoarrow as na
        >>> na.fixed_size_binary(123).byte_width
        123
        """

        if self._c_schema_view.type_id == _types.FIXED_SIZE_BINARY:
            return self._c_schema_view.fixed_size

    @property
    def unit(self) -> Union[TimeUnit, None]:
        """TimeUnit for timestamp, time, and duration types

        Returns ``None`` for types for which this property is not relevant.

        >>> import nanoarrow as na
        >>> na.timestamp(na.TimeUnit.SECOND).unit
        <TimeUnit.SECOND: 0>
        """

        unit_id = self._c_schema_view.time_unit_id
        if unit_id is not None:
            return TimeUnit(unit_id)

    @property
    def timezone(self) -> Union[str, None]:
        """Timezone for timestamp types

        Returns ``None`` for types for which this property is not relevant or
        for timezone types for which the timezone is not set.

        >>> import nanoarrow as na
        >>> na.timestamp(na.TimeUnit.SECOND, timezone="America/Halifax").timezone
        'America/Halifax'
        """
        if self._c_schema_view.timezone:
            return self._c_schema_view.timezone

    @property
    def precision(self) -> int:
        """Decimal precision

        >>> import nanoarrow as na
        >>> na.decimal128(10, 3).precision
        10
        """
        return self._c_schema_view.decimal_precision

    @property
    def scale(self) -> int:
        """Decimal scale

        >>> import nanoarrow as na
        >>> na.decimal128(10, 3).scale
        3
        """

        return self._c_schema_view.decimal_scale

    @property
    def index_type(self) -> Union["Schema", None]:
        """Dictionary index type

        For dictionary types, the type corresponding to the indices.
        See also :attr:`value_type`.

        >>> import nanoarrow as na
        >>> na.dictionary(na.int32(), na.string()).index_type
        <Schema> int32
        """
        if self._c_schema_view.type_id == _types.DICTIONARY:
            index_schema = self._c_schema.modify(
                dictionary=False, flags=0, nullable=self.nullable
            )
            return Schema(index_schema)
        else:
            return None

    @property
    def dictionary_ordered(self) -> Union[bool, None]:
        """Dictionary ordering

        For dictionary types, returns ``True`` if the order of dictionary values
        are meaningful.

        >>> import nanoarrow as na
        >>> na.dictionary(na.int32(), na.string()).dictionary_ordered
        False
        """
        return self._c_schema_view.dictionary_ordered

    @property
    def value_type(self) -> Union["Schema", None]:
        """Dictionary, map, or list value type

        >>> import nanoarrow as na
        >>> na.list_(na.int32()).value_type
        <Schema> 'item': int32
        >>> na.map_(na.int32(), na.string()).value_type
        <Schema> 'value': string
        >>> na.dictionary(na.int32(), na.string()).value_type
        <Schema> string
        """
        if self._c_schema_view.type_id in (
            _types.LIST,
            _types.LARGE_LIST,
            _types.FIXED_SIZE_LIST,
        ):
            return self.field(0)
        elif self._c_schema_view.type_id == _types.MAP:
            return Schema(self._c_schema.child(0).child(1))
        elif self._c_schema_view.type_id == _types.DICTIONARY:
            return Schema(self._c_schema.dictionary)
        else:
            return None

    @property
    def key_type(self) -> Union["Schema", None]:
        """Map key type

        >>> import nanoarrow as na
        >>> na.map_(na.int32(), na.string()).key_type
        <Schema> 'key': non-nullable int32
        """
        if self._c_schema_view.type_id == _types.MAP:
            return Schema(self._c_schema.child(0).child(0))
        else:
            return None

    @property
    def keys_sorted(self) -> Union[bool, None]:
        if self._c_schema_view.type_id == _types.MAP:
            return self._c_schema_view.map_keys_sorted
        else:
            return None

    @property
    def list_size(self) -> Union[int, None]:
        """Fixed-size list element size

        >>> import nanoarrow as na
        >>> na.fixed_size_list(na.int32(), 123).list_size
        123
        """
        if self._c_schema_view.type_id == _types.FIXED_SIZE_LIST:
            return self._c_schema_view.fixed_size
        else:
            return None

    @property
    def n_fields(self) -> int:
        """Number of child Schemas

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.n_fields
        1
        """

        return self._c_schema.n_children

    def field(self, i) -> "Schema":
        """Extract a child Schema

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.field(0)
        <Schema> 'col1': int32
        """

        # Returning a copy to reduce interdependence between Schema instances:
        # The CSchema keeps its parent alive when wrapping a child, which might
        # be unexpected if the parent schema is very large.
        return Schema(self._c_schema.child(i).__deepcopy__())

    @property
    def fields(self) -> List["Schema"]:
        """Iterate over child Schemas

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> for field in schema.fields:
        ...     print(field.name)
        ...
        col1
        """
        return [self.field(i) for i in range(self.n_fields)]

    def serialize(self, dst=None) -> Union[bytes, None]:
        """Write this Schema into dst as an encapsulated IPC message

        Parameters
        ----------
        dst : file-like, optional
            If present, a file-like object into which the schema should be
            serialized. If omitted, this will create a ``io.BytesIO()`` and
            return the serialized result.
        """
        from nanoarrow.c_array_stream import CArrayStream

        from nanoarrow.ipc import StreamWriter

        empty = CArrayStream.from_c_arrays([], self._c_schema, validate=False)

        if dst is None:
            import io

            with io.BytesIO() as dst:
                writer = StreamWriter.from_writable(dst)
                writer.write_stream(empty)
                return dst.getvalue()
        else:
            writer = StreamWriter.from_writable(dst)
            writer.write_stream(empty)

    def __repr__(self) -> str:
        # An empty max_char_width prints the entire schema
        return _schema_repr(self, max_char_width=0)

    def __arrow_c_schema__(self):
        return self._c_schema.__arrow_c_schema__()


def schema(obj, **kwargs) -> Schema:
    """
    Alias for the :class:`Schema` class constructor. The use of
    ``nanoarrow.Schema()`` is preferred over ``nanoarrow.schema()``.
    """
    return Schema(obj, **kwargs)


def null(nullable: bool = True) -> Schema:
    """Create an instance of a null type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.null()
    <Schema> na
    """
    return Schema(Type.NULL, nullable=nullable)


def bool_(nullable: bool = True) -> Schema:
    """Create an instance of a boolean type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.bool_()
    <Schema> bool
    """
    return Schema(Type.BOOL, nullable=nullable)


def int8(nullable: bool = True) -> Schema:
    """Create an instance of a signed 8-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.int8()
    <Schema> int8
    """
    return Schema(Type.INT8, nullable=nullable)


def uint8(nullable: bool = True) -> Schema:
    """Create an instance of an unsigned 8-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.uint8()
    <Schema> uint8
    """
    return Schema(Type.UINT8, nullable=nullable)


def int16(nullable: bool = True) -> Schema:
    """Create an instance of a signed 16-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.int16()
    <Schema> int16
    """
    return Schema(Type.INT16, nullable=nullable)


def uint16(nullable: bool = True) -> Schema:
    """Create an instance of an unsigned 16-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.uint16()
    <Schema> uint16
    """
    return Schema(Type.UINT16, nullable=nullable)


def int32(nullable: bool = True) -> Schema:
    """Create an instance of a signed 32-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.int32()
    <Schema> int32
    """
    return Schema(Type.INT32, nullable=nullable)


def uint32(nullable: bool = True) -> Schema:
    """Create an instance of an unsigned 32-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.uint32()
    <Schema> uint32
    """
    return Schema(Type.UINT32, nullable=nullable)


def int64(nullable: bool = True) -> Schema:
    """Create an instance of a signed 32-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.int64()
    <Schema> int64
    """
    return Schema(Type.INT64, nullable=nullable)


def uint64(nullable: bool = True) -> Schema:
    """Create an instance of an unsigned 32-bit integer type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.uint64()
    <Schema> uint64
    """
    return Schema(Type.UINT64, nullable=nullable)


def float16(nullable: bool = True) -> Schema:
    """Create an instance of a 16-bit floating-point type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.float16()
    <Schema> half_float
    """
    return Schema(Type.HALF_FLOAT, nullable=nullable)


def float32(nullable: bool = True) -> Schema:
    """Create an instance of a 32-bit floating-point type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.float32()
    <Schema> float
    """
    return Schema(Type.FLOAT, nullable=nullable)


def float64(nullable: bool = True) -> Schema:
    """Create an instance of a 64-bit floating-point type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.float64()
    <Schema> double
    """
    return Schema(Type.DOUBLE, nullable=nullable)


def string(nullable: bool = True) -> Schema:
    """Create an instance of a variable-length UTF-8 encoded string type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.string()
    <Schema> string
    """
    return Schema(Type.STRING, nullable=nullable)


def large_string(nullable: bool = True) -> Schema:
    """Create an instance of a variable-length UTF-8 encoded string type
    that uses 64-bit offsets.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.large_string()
    <Schema> large_string
    """
    return Schema(Type.LARGE_STRING, nullable=nullable)


def string_view(nullable: bool = True) -> Schema:
    """Create an instance of a string view type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.string_view()
    <Schema> string_view
    """
    return Schema(Type.STRING_VIEW, nullable=nullable)


def binary(nullable: bool = True) -> Schema:
    """Create an instance of a variable or fixed-width binary type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.binary()
    <Schema> binary
    """
    return Schema(Type.BINARY, nullable=nullable)


def large_binary(nullable: bool = True) -> Schema:
    """Create an instance of a variable-length binary type that uses 64-bit offsets.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.large_binary()
    <Schema> large_binary
    """
    return Schema(Type.LARGE_BINARY, nullable=nullable)


def binary_view(nullable: bool = True) -> Schema:
    """Create an instance of a binary view type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.binary_view()
    <Schema> binary_view
    """
    return Schema(Type.BINARY_VIEW, nullable=nullable)


def fixed_size_binary(byte_width: int, nullable: bool = True) -> Schema:
    """Create an instance of a variable or fixed-width binary type.

    Parameters
    ----------
    byte_width : int
        The width of each element in bytes.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.fixed_size_binary(123)
    <Schema> fixed_size_binary(123)
    """
    return Schema(Type.FIXED_SIZE_BINARY, byte_width=byte_width, nullable=nullable)


def date32(nullable: bool = True) -> Schema:
    """Create an instance of a 32-bit date type (days since 1970-01-01).

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.date32()
    <Schema> date32
    """
    return Schema(Type.DATE32, nullable=nullable)


def date64(nullable: bool = True) -> Schema:
    """Create an instance of a 64-bit date type (milliseconds since 1970-01-01).

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.date64()
    <Schema> date64
    """
    return Schema(Type.DATE64, nullable=nullable)


def time32(unit: Union[str, TimeUnit], nullable: bool = True) -> Schema:
    """Create an instance of a 32-bit time of day type.

    Parameters
    ----------
    unit : str or :class:`TimeUnit`
        The unit of values stored by this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.time32("s")
    <Schema> time32('s')
    """
    return Schema(Type.TIME32, unit=unit, nullable=nullable)


def time64(unit: Union[str, TimeUnit], nullable: bool = True) -> Schema:
    """Create an instance of a 64-bit time of day type.

    Parameters
    ----------
    unit : str or :class:`TimeUnit`
        The unit of values stored by this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.time64("us")
    <Schema> time64('us')
    """
    return Schema(Type.TIME64, unit=unit, nullable=nullable)


def timestamp(
    unit: Union[str, TimeUnit], timezone: Union[str, None] = None, nullable: bool = True
) -> Schema:
    """Create an instance of a timestamp type.

    Parameters
    ----------
    unit : str or :class:`TimeUnit`
        The unit of values stored by this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.timestamp("s")
    <Schema> timestamp('s', '')
    >>> na.timestamp("s", timezone="America/Halifax")
    <Schema> timestamp('s', 'America/Halifax')
    """
    return Schema(Type.TIMESTAMP, timezone=timezone, unit=unit, nullable=nullable)


def duration(unit, nullable: bool = True) -> Schema:
    """Create an instance of a duration type.

    Parameters
    ----------
    unit : str or :class:`TimeUnit`
        The unit of values stored by this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.duration("s")
    <Schema> duration('s')
    """
    return Schema(Type.DURATION, unit=unit, nullable=nullable)


def interval_months(nullable: bool = True) -> Schema:
    """Create an instance of an interval type measured in months.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.interval_months()
    <Schema> interval_months
    """
    return Schema(Type.INTERVAL_MONTHS, nullable=nullable)


def interval_day_time(nullable: bool = True) -> Schema:
    """Create an instance of an interval type measured as a day/time pair.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.interval_day_time()
    <Schema> interval_day_time
    """
    return Schema(Type.INTERVAL_DAY_TIME, nullable=nullable)


def interval_month_day_nano(nullable: bool = True) -> Schema:
    """Create an instance of an interval type measured as a month/day/nanosecond
    tuple.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.interval_month_day_nano()
    <Schema> interval_month_day_nano
    """
    return Schema(Type.INTERVAL_MONTH_DAY_NANO, nullable=nullable)


def decimal128(precision: int, scale: int, nullable: bool = True) -> Schema:
    """Create an instance of a 128-bit decimal type.

    Parameters
    ----------
    precision : int
        The number of significant digits representable by this type. Must be
        between 1 and 38.
    scale : int
        The number of digits after the decimal point for values of this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.decimal128(10, 3)
    <Schema> decimal128(10, 3)
    """
    return Schema(Type.DECIMAL128, precision=precision, scale=scale, nullable=nullable)


def decimal256(precision: int, scale: int, nullable: bool = True) -> Schema:
    """Create an instance of a 256-bit decimal type.

    Parameters
    ----------
    precision : int
        The number of significant digits representable by this type. Must be
        between 1 and 76.
    scale : int
        The number of digits after the decimal point for values of this type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.decimal256(10, 3)
    <Schema> decimal256(10, 3)
    """
    return Schema(Type.DECIMAL256, precision=precision, scale=scale, nullable=nullable)


def struct(fields, nullable: bool = True) -> Schema:
    """Create a type representing a named sequence of fields.

    Parameters
    ----------
    fields :
        * A dictionary whose keys are field names and values are schema-like objects
        * An iterable whose items are a schema like objects where the field name is
          inherited from the schema-like object.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.struct([na.int32()])
    <Schema> struct<: int32>
    >>> na.struct({"col1": na.int32()})
    <Schema> struct<col1: int32>
    """
    return Schema(Type.STRUCT, fields=fields, nullable=nullable)


def list_(value_type, nullable: bool = True) -> Schema:
    """Create a type representing a variable-size list of some other type.

    Parameters
    ----------
    value_type : schema-like
        The type of values in each list element.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.list_(na.int32())
    <Schema> list<item: int32>
    """
    return Schema(Type.LIST, value_type=value_type, nullable=nullable)


def large_list(value_type, nullable: bool = True) -> Schema:
    """Create a type representing a variable-size list of some other type.

    Unlike :func:`list_`, the func:`large_list` can accomodate arrays
    with more than ``2 ** 31 - 1`` items in the values array.

    Parameters
    ----------
    value_type : schema-like
        The type of values in each list element.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.large_list(na.int32())
    <Schema> large_list<item: int32>
    """
    return Schema(Type.LARGE_LIST, value_type=value_type, nullable=nullable)


def fixed_size_list(value_type, list_size: int, nullable: bool = True) -> Schema:
    """Create a type representing a fixed-size list of some other type.

    Parameters
    ----------
    value_type : schema-like
        The type of values in each list element.
    list_size : int
        The number of values in each list element.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.fixed_size_list(na.int32(), 123)
    <Schema> fixed_size_list(123)<item: int32>
    """
    return Schema(
        Type.FIXED_SIZE_LIST,
        value_type=value_type,
        list_size=list_size,
        nullable=nullable,
    )


def map_(key_type, value_type, keys_sorted: bool = False, nullable: bool = True):
    """Create a type representing a list of key/value mappings

    Note that each element in the list contains potentially many
    key/value pairs (and that a map array contains potentially
    many individual mappings).

    Parameters
    ----------
    value_type : schema-like
        The type of keys in each map element.
    value_type : schema-like
        The type of values in each map element
    keys_sorted : bool, optional
        True if keys within each map element are sorted.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------
    >>> import nanoarrow as na
    >>> na.map_(na.int32(), na.string())
    <Schema> map<entries: struct<key: int32, value: string>>
    """
    return Schema(
        Type.MAP,
        key_type=key_type,
        value_type=value_type,
        keys_sorted=keys_sorted,
        nullable=nullable,
    )


def dictionary(index_type, value_type, dictionary_ordered: bool = False) -> Schema:
    """Create a type representing dictionary-encoded values

    Parameters
    ----------
    index_type : schema-like
        The data type of the indices. Must be an integral type.
    value_type : schema-like
        The type of the dictionary array.
    ordered: bool, optional
        Use ``True`` if the order of values in the dictionary array is
        meaningful.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.dictionary(na.int32(), na.string())
    <Schema> dictionary(int32)<string>
    """
    return Schema(
        Type.DICTIONARY,
        index_type=index_type,
        value_type=value_type,
        dictionary_ordered=dictionary_ordered,
    )


def extension_type(
    storage_schema,
    extension_name: str,
    extension_metadata: Union[str, bytes, None] = None,
    nullable: bool = True,
) -> Schema:
    """Create an Arrow extension type

    Parameters
    ----------
    extension_name: str
        The extension name to associate with this type.
    extension_metadata: str or bytes, optional
        Extension metadata containing extension parameters associated with this
        extension type.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.
    """
    storage_schema = c_schema(storage_schema)
    storage_metadata = storage_schema.metadata
    metadata = dict(storage_metadata) if storage_metadata else {}
    metadata["ARROW:extension:name"] = extension_name
    if extension_metadata:
        metadata["ARROW:extension:metadata"] = extension_metadata
    else:
        metadata["ARROW:extension:metadata"] = ""

    return Schema(storage_schema, nullable=nullable, metadata=metadata)


def _c_schema_from_type_and_params(type: Type, params: dict):
    factory = CSchemaBuilder.allocate()

    if _types.is_decimal(type.value):
        precision = int(params.pop("precision"))
        scale = int(params.pop("scale"))
        factory.set_type_decimal(type.value, precision, scale)

    elif _types.has_time_unit(type.value):
        time_unit = params.pop("unit")
        if "timezone" in params:
            timezone = params.pop("timezone")
        else:
            timezone = None

        factory.set_type_date_time(
            type.value, TimeUnit.create(time_unit).value, timezone
        )

    elif type == Type.FIXED_SIZE_BINARY:
        factory.set_type_fixed_size(type.value, int(params.pop("byte_width")))

    elif type == Type.LIST:
        factory.set_format("+l")
        factory.allocate_children(1)
        factory.set_child(0, "item", c_schema(params.pop("value_type")))

    elif type == Type.LARGE_LIST:
        factory.set_format("+L")
        factory.allocate_children(1)
        factory.set_child(0, "item", c_schema(params.pop("value_type")))

    elif type == Type.FIXED_SIZE_LIST:
        fixed_size = int(params.pop("list_size"))
        factory.set_format(f"+w:{fixed_size}")
        factory.allocate_children(1)
        factory.set_child(0, "item", c_schema(params.pop("value_type")))

    elif type == Type.MAP:
        key_schema = c_schema(params.pop("key_type"))
        value_schema = c_schema(params.pop("value_type"))

        entries = CSchemaBuilder.allocate()
        entries.set_format("+s")
        entries.set_nullable(False)
        entries.allocate_children(2)
        entries.set_child(0, "key", key_schema.modify(nullable=False))
        entries.set_child(1, "value", value_schema)

        factory.set_format("+m")
        factory.allocate_children(1)
        factory.set_child(0, "entries", entries.finish())
        factory.set_nullable(False)

        if "keys_sorted" in params:
            factory.set_map_keys_sorted(params.pop("keys_sorted"))

    elif type == Type.DICTIONARY:
        index_type = c_schema(params.pop("index_type"))
        factory.set_format(index_type.format)

        value_type = c_schema(params.pop("value_type"))
        factory.set_dictionary(value_type)

        if "dictionary_ordered" in params and bool(params.pop("dictionary_ordered")):
            factory.set_dictionary_ordered(True)

    else:
        factory.set_type(type.value)

    if params:
        unused = ", ".join(f"'{item}'" for item in params.keys())
        raise ValueError(f"Unused parameters whilst constructing Schema: {unused}")

    # Better default than NULL, which causes some implementations to crash
    factory.set_name("")

    return factory.finish()


def _clean_fields(fields):
    if fields is None:
        return None
    elif hasattr(fields, "items"):
        return {k: c_schema(v) for k, v in fields.items()}
    else:
        return [c_schema(v) for v in fields]


def _schema_repr(obj, max_char_width=80, prefix="<Schema> ", include_metadata=True):
    lines = []

    modifiers = []

    if obj.name:
        name = reprlib.Repr().repr(obj.name)
        modifiers.append(f"{name}:")

    if not obj.nullable:
        modifiers.append("non-nullable")

    if obj.dictionary_ordered:
        modifiers.append("ordered")

    # Ensure extra space at the end of the modifiers
    modifiers.append("")

    modifiers_str = " ".join(modifiers)
    first_line_prefix = f"{prefix}{modifiers_str}"
    max_char_width = max(max_char_width - len(first_line_prefix), 0)

    schema_str = _repr_utils.c_schema_to_string(obj._c_schema, max_char_width)
    lines.append(f"{first_line_prefix}{schema_str}")

    if include_metadata:
        metadata_dict = dict(obj.metadata.items())
        if metadata_dict:
            metadata_dict_repr = reprlib.Repr().repr(metadata_dict)
            metadata_line = f"- metadata: {metadata_dict_repr[:max_char_width]}"
            lines.append(metadata_line[:max_char_width])

    return "\n".join(lines)


_PARAM_NAMES = {
    _types.FIXED_SIZE_BINARY: ("byte_width",),
    _types.TIMESTAMP: ("unit", "timezone"),
    _types.TIME32: ("unit",),
    _types.TIME64: ("unit",),
    _types.DURATION: ("unit",),
    _types.DECIMAL128: ("precision", "scale"),
    _types.DECIMAL256: ("precision", "scale"),
    _types.STRUCT: ("fields",),
    _types.LIST: ("value_type",),
    _types.LARGE_LIST: ("value_type",),
    _types.FIXED_SIZE_LIST: ("value_type", "list_size"),
    _types.DICTIONARY: ("index_type", "value_type", "dictionary_ordered"),
}
