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
from typing import Union

from nanoarrow._lib import CArrowTimeUnit, CArrowType, CSchemaBuilder, CSchemaView
from nanoarrow.c_lib import c_schema


class Type(enum.Enum):
    """The Type enumerator provides a means by which the various type
    categories can be identified. Type values can be used in place of
    :class:`Schema` instances in most places for parameter-free types.
    """

    UNINITIALIZED = CArrowType.UNINITIALIZED
    NULL = CArrowType.NA
    BOOL = CArrowType.BOOL
    UINT8 = CArrowType.UINT8
    INT8 = CArrowType.INT8
    UINT16 = CArrowType.UINT16
    INT16 = CArrowType.INT16
    UINT32 = CArrowType.UINT32
    INT32 = CArrowType.INT32
    UINT64 = CArrowType.UINT64
    INT64 = CArrowType.INT64
    HALF_FLOAT = CArrowType.HALF_FLOAT
    FLOAT = CArrowType.FLOAT
    DOUBLE = CArrowType.DOUBLE
    STRING = CArrowType.STRING
    BINARY = CArrowType.BINARY
    FIXED_SIZE_BINARY = CArrowType.FIXED_SIZE_BINARY
    DATE32 = CArrowType.DATE32
    DATE64 = CArrowType.DATE64
    TIMESTAMP = CArrowType.TIMESTAMP
    TIME32 = CArrowType.TIME32
    TIME64 = CArrowType.TIME64
    INTERVAL_MONTHS = CArrowType.INTERVAL_MONTHS
    INTERVAL_DAY_TIME = CArrowType.INTERVAL_DAY_TIME
    DECIMAL128 = CArrowType.DECIMAL128
    DECIMAL256 = CArrowType.DECIMAL256
    LIST = CArrowType.LIST
    STRUCT = CArrowType.STRUCT
    SPARSE_UNION = CArrowType.SPARSE_UNION
    DENSE_UNION = CArrowType.DENSE_UNION
    DICTIONARY = CArrowType.DICTIONARY
    MAP = CArrowType.MAP
    EXTENSION = CArrowType.EXTENSION
    FIXED_SIZE_LIST = CArrowType.FIXED_SIZE_LIST
    DURATION = CArrowType.DURATION
    LARGE_STRING = CArrowType.LARGE_STRING
    LARGE_BINARY = CArrowType.LARGE_BINARY
    LARGE_LIST = CArrowType.LARGE_LIST
    INTERVAL_MONTH_DAY_NANO = CArrowType.INTERVAL_MONTH_DAY_NANO

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


class Schema:
    """The Schema is nanoarrow's high-level data type representation whose scope maps to
    that of the ArrowSchema in the Arrow C Data interface. See :func:`schema` for class
    details.
    """

    def __init__(
        self,
        obj,
        *,
        name=None,
        nullable=None,
        **params,
    ) -> None:
        if isinstance(obj, Type):
            self._c_schema = _c_schema_from_type_and_params(obj, params, name, nullable)
        elif not params and nullable is None and name is None:
            self._c_schema = c_schema(obj)
        else:
            # A future version could also deep copy the schema and update it if these
            # values *are* specified.
            raise ValueError(
                "params, nullable, and name must be unspecified if type is not "
                "nanoarrow.Type"
            )

        self._c_schema_view = CSchemaView(self._c_schema)

    @property
    def type(self) -> Type:
        """Type enumerator value of this Schema

        >>> import nanoarrow as na
        >>> na.int32().type
        <Type.INT32: 8>
        """
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

    @property
    def byte_width(self) -> Union[int, None]:
        """Element byte width for fixed-size binary type

        Returns ``None`` for types for which this property is not relevant.

        >>> import nanoarrow as na
        >>> na.fixed_size_binary(123).byte_width
        123
        """

        if self._c_schema_view.type_id == CArrowType.FIXED_SIZE_BINARY:
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
    def n_fields(self) -> int:
        """Number of child Schemas

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.n_fields
        1
        """

        return self._c_schema.n_children

    def field(self, i):
        """Extract a child Schema

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.field(0)
        Schema(INT32, name='col1')
        """

        # Returning a copy to reduce interdependence between Schema instances:
        # The CSchema keeps its parent alive when wrapping a child, which might
        # be unexpected if the parent schema is very large.
        return Schema(self._c_schema.child(i).__deepcopy__())

    @property
    def fields(self):
        """Iterate over child Schemas

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> for field in schema.fields:
        ...     print(field.name)
        ...
        col1
        """
        for i in range(self.n_fields):
            yield self.field(i)

    def __repr__(self) -> str:
        return _schema_repr(self)

    def __arrow_c_schema__(self):
        return self._c_schema.__arrow_c_schema__()


def schema(obj, *, name=None, nullable=None, **params):
    """Create a nanoarrow Schema

    The Schema is nanoarrow's high-level data type representation, encompasing
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
        Only supported if ``obj`` is a :class:`Type` object (for any other input,
        the nullability is preserved from the passed object).

    params :
        Type-specific parameters when ``obj`` is a :class:`Type`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> import pyarrow as pa
    >>> na.schema(na.Type.INT32)
    Schema(INT32)
    >>> na.schema(na.Type.DURATION, unit=na.TimeUnit.SECOND)
    Schema(DURATION, unit=SECOND)
    >>> na.schema(pa.int32())
    Schema(INT32)
    """
    return Schema(obj, name=name, nullable=nullable, **params)


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
    Schema(NULL)
    """
    return Schema(Type.NULL, nullable=nullable)


def bool(nullable: bool = True) -> Schema:
    """Create an instance of a boolean type.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.bool()
    Schema(BOOL)
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
    Schema(INT8)
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
    Schema(UINT8)
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
    Schema(INT16)
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
    Schema(UINT16)
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
    Schema(INT32)
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
    Schema(UINT32)
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
    Schema(INT64)
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
    Schema(UINT64)
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
    Schema(HALF_FLOAT)
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
    Schema(FLOAT)
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
    Schema(DOUBLE)
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
    Schema(STRING)
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
    Schema(LARGE_STRING)
    """
    return Schema(Type.LARGE_STRING, nullable=nullable)


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
    Schema(BINARY)
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
    Schema(LARGE_BINARY)
    """
    return Schema(Type.LARGE_BINARY, nullable=nullable)


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
    Schema(FIXED_SIZE_BINARY, byte_width=123)
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
    Schema(DATE32)
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
    Schema(DATE64)
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
    Schema(TIME32, unit=SECOND)
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
    Schema(TIME64, unit=MICRO)
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
    Schema(TIMESTAMP, unit=SECOND)
    >>> na.timestamp("s", timezone="America/Halifax")
    Schema(TIMESTAMP, unit=SECOND, timezone='America/Halifax')
    """
    return Schema(Type.TIMESTAMP, timezone=timezone, unit=unit, nullable=nullable)


def duration(unit, nullable: bool = True):
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
    Schema(DURATION, unit=SECOND)
    """
    return Schema(Type.DURATION, unit=unit, nullable=nullable)


def interval_months(nullable: bool = True):
    """Create an instance of an interval type measured in months.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.interval_months()
    Schema(INTERVAL_MONTHS)
    """
    return Schema(Type.INTERVAL_MONTHS, nullable=nullable)


def interval_day_time(nullable: bool = True):
    """Create an instance of an interval type measured as a day/time pair.

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.interval_day_time()
    Schema(INTERVAL_DAY_TIME)
    """
    return Schema(Type.INTERVAL_DAY_TIME, nullable=nullable)


def interval_month_day_nano(nullable: bool = True):
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
    Schema(INTERVAL_MONTH_DAY_NANO)
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
    Schema(DECIMAL128, precision=10, scale=3)
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
    Schema(DECIMAL256, precision=10, scale=3)
    """
    return Schema(Type.DECIMAL256, precision=precision, scale=scale, nullable=nullable)


def struct(fields, nullable=True) -> Schema:
    """Create a type representing a named sequence of fields.

    Parameters
    ----------
    fields :
        * A dictionary whose keys are field names and values are schema-like objects
        * An iterable whose items are a schema like object or a two-tuple of the
          field name and a schema-like object. If a field name is not specified
          from the tuple, the field name is inherited from the schema-like object.
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.

    Examples
    --------

    >>> import nanoarrow as na
    >>> na.struct([na.int32()])
    Schema(STRUCT, fields=[Schema(INT32)])
    >>> na.struct([("col1", na.int32())])
    Schema(STRUCT, fields=[Schema(INT32, name='col1')])
    >>> na.struct({"col1": na.int32()})
    Schema(STRUCT, fields=[Schema(INT32, name='col1')])
    """
    return Schema(Type.STRUCT, fields=fields, nullable=nullable)


def _c_schema_from_type_and_params(
    type: Type,
    params: dict,
    name: Union[bool, None, bool],
    nullable: Union[bool, None],
):
    factory = CSchemaBuilder.allocate()

    if type == Type.STRUCT:
        fields = _clean_fields(params.pop("fields"))

        factory.set_format("+s").allocate_children(len(fields))
        for i, item in enumerate(fields):
            child_name, c_schema = item
            factory.set_child(i, child_name, c_schema)

    elif type.value in CSchemaView._decimal_types:
        precision = int(params.pop("precision"))
        scale = int(params.pop("scale"))
        factory.set_type_decimal(type.value, precision, scale)

    elif type.value in CSchemaView._time_unit_types:
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

    else:
        factory.set_type(type.value)

    if params:
        unused = ", ".join(f"'{item}'" for item in params.keys())
        raise ValueError(f"Unused parameters whilst constructing Schema: {unused}")

    # Apply default nullability (True)
    if nullable is None:
        nullable = True
    factory.set_nullable(nullable)

    # Apply default name (an empty string). To explicitly set a NULL
    # name, a caller would have to specify False.
    if name is None:
        name = ""
    elif name is False:
        name = None
    factory.set_name(name)

    return factory.finish()


def _clean_fields(fields):
    if isinstance(fields, dict):
        return [(str(k), c_schema(v)) for k, v in fields.items()]
    else:
        fields_clean = []
        for item in fields:
            if isinstance(item, tuple) and len(item) == 2:
                fields_clean.append((str(item[0]), c_schema(item[1])))
            else:
                fields_clean.append((None, c_schema(item)))

        return fields_clean


def _schema_repr(obj):
    out = f"Schema({_schema_param_repr('type', obj.type)}"

    if obj.name is None:
        out += ", name=False"
    elif obj.name:
        out += f", name={_schema_param_repr('name', obj.name)}"

    if obj._c_schema_view.type_id not in _PARAM_NAMES:
        param_names = []
    else:
        param_names = _PARAM_NAMES[obj._c_schema_view.type_id]

    for name in param_names:
        value = getattr(obj, name)
        if value is None:
            continue
        out += ", "
        param_repr = f"{name}={_schema_param_repr(name, getattr(obj, name))}"
        out += param_repr

    if not obj.nullable:
        out += ", nullable=False"

    out += ")"
    return out


def _schema_param_repr(name, value):
    if name == "type":
        return f"{value.name}"
    elif name == "unit":
        return f"{value.name}"
    elif name == "fields":
        # It would be nice to indent this/get it on multiple lines since
        # most output will be uncomfortably wide even with the abbreviated repr
        return reprlib.Repr().repr(list(value))
    else:
        return reprlib.Repr().repr(value)


_PARAM_NAMES = {
    CArrowType.FIXED_SIZE_BINARY: ("byte_width",),
    CArrowType.TIMESTAMP: ("unit", "timezone"),
    CArrowType.TIME32: ("unit",),
    CArrowType.TIME64: ("unit",),
    CArrowType.DURATION: ("unit",),
    CArrowType.DECIMAL128: ("precision", "scale"),
    CArrowType.DECIMAL256: ("precision", "scale"),
    CArrowType.STRUCT: ("fields",),
}
