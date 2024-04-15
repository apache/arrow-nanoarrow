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

import warnings
from functools import cached_property
from itertools import islice
from typing import Iterable, Tuple

from nanoarrow.c_lib import (
    CArrayView,
    CArrowType,
    c_array_stream,
    c_schema,
    c_schema_view,
)


def iter_py(obj, schema=None) -> Iterable:
    """Iterate over items in zero or more arrays

    Returns an iterator over an array stream where each item is a
    Python representation of the next element.

    Paramters
    ---------
    obj : array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow import iterator
    >>> array = na.c_array([1, 2, 3], na.int32())
    >>> list(iterator.iter_py(array))
    [1, 2, 3]
    """
    return PyIterator.get_iterator(obj, schema=schema)


def iter_tuples(obj, schema=None) -> Iterable[Tuple]:
    """Iterate over rows in zero or more struct arrays

    Returns an iterator over an array stream of struct arrays (i.e.,
    record batches) where each item is a tuple of the items in each
    row. This is different than :func:`iter_py`, which encodes struct
    columns as dictionaries.

    Paramters
    ---------
    obj : array stream-like
        An array-like or array stream-like object as sanitized by
        :func:`c_array_stream`.
    schema : schema-like, optional
        An optional schema, passed to :func:`c_array_stream`.

    Examples
    --------

    >>> import nanoarrow as na
    >>> from nanoarrow import iterator
    >>> import pyarrow as pa
    >>> array = pa.record_batch([pa.array([1, 2, 3])], names=["col1"])
    >>> list(iterator.iter_tuples(array))
    [(1,), (2,), (3,)]
    """
    return RowTupleIterator.get_iterator(obj, schema=schema)


class InvalidArrayWarning(UserWarning):
    pass


class LossyConversionWarning(UserWarning):
    pass


class ArrayViewIterator:
    """Base class for iterators that use an internal ArrowArrayView
    as the basis for conversion to Python objects. Intended for internal use.
    """

    def __init__(self, schema, *, _array_view=None):
        self._schema = c_schema(schema)
        self._schema_view = c_schema_view(schema)

        if _array_view is None:
            self._array_view = CArrayView.from_schema(self._schema)
        else:
            self._array_view = _array_view

        self._children = list(
            map(self._make_child, self._schema.children, self._array_view.children)
        )

        if self._schema.dictionary is None:
            self._dictionary = None
        else:
            self._dictionary = self._make_child(
                self._schema.dictionary, self._array_view.dictionary
            )

    def _make_child(self, schema, array_view):
        return type(self)(schema, _array_view=array_view)

    @cached_property
    def _child_names(self):
        return [child.name for child in self._schema.children]

    @cached_property
    def _object_label(self):
        if self._schema.name:
            return f"{self._schema.name} <{self._schema_view.type}>"
        else:
            return f"<unnamed {self._schema_view.type}>"

    def _contains_nulls(self):
        return (
            self._schema_view.nullable
            and len(self._array_view.buffer(0))
            and self._array_view.null_count != 0
        )

    def _set_array(self, array):
        self._array_view._set_array(array)
        return self

    def _warn(self, message, category):
        warnings.warn(f"{self._object_label}: {message}", category)


class PyIterator(ArrayViewIterator):
    """Iterate over the Python object version of values in an ArrowArrayView.
    Intended for internal use.
    """

    @classmethod
    def get_iterator(cls, obj, schema=None):
        with c_array_stream(obj, schema=schema) as stream:
            iterator = cls(stream._get_cached_schema())
            for array in stream:
                iterator._set_array(array)
                yield from iterator._iter1(0, array.length)

    def _iter1(self, offset, length):
        type_id = self._schema_view.type_id
        if type_id not in _ITEMS_ITER_LOOKUP:
            raise KeyError(
                f"Can't resolve iterator for type '{self._schema_view.type}'"
            )

        factory = getattr(self, _ITEMS_ITER_LOOKUP[type_id])
        return factory(offset, length)

    def _dictionary_iter(self, offset, length):
        dictionary = list(
            self._dictionary._iter1(0, self._dictionary._array_view.length)
        )
        for dict_index in self._primitive_iter(offset, length):
            yield None if dict_index is None else dictionary[dict_index]

    def _wrap_iter_nullable(self, validity, items):
        for is_valid, item in zip(validity, items):
            yield item if is_valid else None

    def _struct_tuple_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        items = zip(*(child._iter1(offset, length) for child in self._children))

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            return self._wrap_iter_nullable(validity, items)
        else:
            return items

    def _struct_iter(self, offset, length):
        names = self._child_names
        for item in self._struct_tuple_iter(offset, length):
            yield None if item is None else {key: val for key, val in zip(names, item)}

    def _list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset

        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]
        child = self._children[0]
        child_iter = child._iter1(starts[0], ends[-1] - starts[0])

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid, start, end in zip(validity, starts, ends):
                item = list(islice(child_iter, end - start))
                yield item if is_valid else None
        else:
            for start, end in zip(starts, ends):
                yield list(islice(child_iter, end - start))

    def _fixed_size_list_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        child = self._children[0]
        fixed_size = view.layout.child_size_elements
        child_iter = child._iter1(offset * fixed_size, length * fixed_size)

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid in validity:
                item = list(islice(child_iter, fixed_size))
                yield item if is_valid else None
        else:
            for _ in range(length):
                yield list(islice(child_iter, fixed_size))

    def _string_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]
        data = memoryview(view.buffer(2))

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid, start, end in zip(validity, starts, ends):
                yield str(data[start:end], "UTF-8") if is_valid else None
        else:
            for start, end in zip(starts, ends):
                yield str(data[start:end], "UTF-8")

    def _binary_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        offsets = memoryview(view.buffer(1))[offset : (offset + length + 1)]
        starts = offsets[:-1]
        ends = offsets[1:]
        data = memoryview(view.buffer(2))

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            for is_valid, start, end in zip(validity, starts, ends):
                yield bytes(data[start:end]) if is_valid else None
        else:
            for start, end in zip(starts, ends):
                yield bytes(data[start:end])

    def _decimal_iter(self, offset, length):
        from decimal import Context, Decimal
        from sys import byteorder

        storage = self._primitive_iter(offset, length)
        precision = self._schema_view.decimal_precision

        # The approach here it to use Decimal(<integer>).scaleb(-scale),
        # which is a balance between simplicity, performance, and
        # safety (ensuring that we stay independent from the global precision).
        # We cache the scaleb and context to avoid doing so in the loop (the
        # argument to scaleb is transformed to a decimal by the .scaleb()
        # implementation).
        #
        # It would probably be fastest to go straight from binary
        # to string to decimal, since creating a decimal from a string
        # appears to be the fastest constructor.
        scaleb = Decimal(-self._schema_view.decimal_scale)
        context = Context(prec=precision)

        for item in storage:
            if item is None:
                yield None
            else:
                int_value = int.from_bytes(item, byteorder)
                yield Decimal(int_value).scaleb(scaleb, context)

    def _date_iter(self, offset, length):
        from datetime import date, timedelta

        storage = self._primitive_iter(offset, length)
        epoch = date(1970, 1, 1)

        if self._schema_view.type_id == CArrowType.DATE32:
            for item in storage:
                if item is None:
                    yield item
                else:
                    yield epoch + timedelta(item)
        else:
            for item in storage:
                if item is None:
                    yield item
                else:
                    yield epoch + timedelta(milliseconds=item)

    def _time_iter(self, offset, length):
        from datetime import time

        for item in self._iter_time_components(offset, length):
            if item is None:
                yield None
            else:
                days, hours, mins, secs, us = item
                if days != 0:
                    self._warn("days != 0", InvalidArrayWarning)

                yield time(hours, mins, secs, us)

    def _timestamp_iter(self, offset, length):
        from datetime import datetime

        epoch = datetime(1970, 1, 1, tzinfo=_get_tzinfo("UTC"))
        parent = self._duration_iter(offset, length)

        tz = self._schema_view.timezone
        if tz:
            tz = _get_tzinfo(tz)

            for item in parent:
                if item is None:
                    yield None
                else:
                    yield (epoch + item).astimezone(tz)
        else:
            epoch = epoch.replace(tzinfo=None)
            for item in parent:
                if item is None:
                    yield None
                else:
                    yield epoch + item

    def _duration_iter(self, offset, length):
        from datetime import timedelta

        storage = self._primitive_iter(offset, length)

        unit = self._schema_view.time_unit
        if unit == "s":
            to_us = 1_000_000
        elif unit == "ms":
            to_us = 1000
        elif unit == "us":
            to_us = 1
        elif unit == "ns":
            storage = self._iter_us_from_ns(storage)
            to_us = 1

        for item in storage:
            if item is None:
                yield None
            else:
                yield timedelta(microseconds=item * to_us)

    def _iter_time_components(self, offset, length):
        storage = self._primitive_iter(offset, length)

        unit = self._schema_view.time_unit
        if unit == "s":
            to_us = 1_000_000
        elif unit == "ms":
            to_us = 1000
        elif unit == "us":
            to_us = 1
        elif unit == "ns":
            storage = self._iter_us_from_ns(storage)
            to_us = 1

        us_per_sec = 1_000_000
        us_per_min = us_per_sec * 60
        us_per_hour = us_per_min * 60
        us_per_day = us_per_hour * 24

        for item in storage:
            if item is None:
                yield None
            else:
                us = item * to_us
                days = us // us_per_day
                us = us % us_per_day
                hours = us // us_per_hour
                us = us % us_per_hour
                mins = us // us_per_min
                us = us % us_per_min
                secs = us // us_per_sec
                us = us % us_per_sec
                yield days, hours, mins, secs, us

    def _iter_us_from_ns(self, parent):
        for item in parent:
            if item is None:
                yield None
            else:
                if item % 1000 != 0:
                    self._warn("nanoseconds discarded", LossyConversionWarning)
                yield item // 1000

    def _primitive_iter(self, offset, length):
        view = self._array_view
        offset += view.offset
        items = view.buffer(1).elements(offset, length)

        if self._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            return self._wrap_iter_nullable(validity, items)
        else:
            return iter(items)


class RowTupleIterator(PyIterator):
    """Iterate over rows of a struct array (stream) where each row is a
    tuple instead of a dictionary. This is usually faster and matches other
    Python concepts more closely (e.g., dbapi's cursor, pandas itertuples).
    Intended for internal use.
    """

    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        if self._schema_view.type != "struct":
            raise TypeError(
                "RowTupleIterator can only iterate over struct arrays "
                f"(got '{self._schema_view.type}')"
            )

    def _make_child(self, schema, array_view):
        return PyIterator(schema, _array_view=array_view)

    def _iter1(self, offset, length):
        return self._struct_tuple_iter(offset, length)


def _get_tzinfo(tz_string, strategy=None):
    import re
    from datetime import timedelta, timezone

    # We can handle UTC without any imports
    if tz_string.upper() == "UTC":
        return timezone.utc

    # Arrow also allows fixed-offset in the from +HH:MM
    maybe_fixed_offset = re.search(r"^([+-])([0-9]{2}):([0-9]{2})$", tz_string)
    if maybe_fixed_offset:
        sign, hours, minutes = maybe_fixed_offset.groups()
        sign = 1 if sign == "+" else -1
        return timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))

    # Try zoneinfo.ZoneInfo() (Python 3.9+)
    if strategy is None or "zoneinfo" in strategy:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(tz_string)
        except ImportError:
            pass

    # Try dateutil.tz.gettz()
    if strategy is None or "dateutil" in strategy:
        try:
            from dateutil.tz import gettz

            return gettz(tz_string)
        except ImportError:
            pass

    raise RuntimeError(
        "zoneinfo (Python 3.9+) or dateutil is required to resolve timezone"
    )


_ITEMS_ITER_LOOKUP = {
    CArrowType.BINARY: "_binary_iter",
    CArrowType.LARGE_BINARY: "_binary_iter",
    CArrowType.STRING: "_string_iter",
    CArrowType.LARGE_STRING: "_string_iter",
    CArrowType.STRUCT: "_struct_iter",
    CArrowType.LIST: "_list_iter",
    CArrowType.LARGE_LIST: "_list_iter",
    CArrowType.FIXED_SIZE_LIST: "_fixed_size_list_iter",
    CArrowType.DICTIONARY: "_dictionary_iter",
    CArrowType.DATE32: "_date_iter",
    CArrowType.DATE64: "_date_iter",
    CArrowType.TIME32: "_time_iter",
    CArrowType.TIME64: "_time_iter",
    CArrowType.TIMESTAMP: "_timestamp_iter",
    CArrowType.DURATION: "_duration_iter",
    CArrowType.DECIMAL128: "_decimal_iter",
    CArrowType.DECIMAL256: "_decimal_iter",
}

_PRIMITIVE_TYPE_NAMES = [
    "BOOL",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "UINT32",
    "INT32",
    "UINT64",
    "INT64",
    "HALF_FLOAT",
    "FLOAT",
    "DOUBLE",
    "FIXED_SIZE_BINARY",
    "INTERVAL_MONTHS",
    "INTERVAL_DAY_TIME",
    "INTERVAL_MONTH_DAY_NANO",
]

for type_name in _PRIMITIVE_TYPE_NAMES:
    type_id = getattr(CArrowType, type_name)
    _ITEMS_ITER_LOOKUP[type_id] = "_primitive_iter"
