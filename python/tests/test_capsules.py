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

import pyarrow as pa

import nanoarrow as na


class SchemaWrapper:
    def __init__(self, schema):
        self.schema = schema

    def __arrow_c_schema__(self):
        return self.schema.__arrow_c_schema__()


class ArrayWrapper:
    def __init__(self, array):
        self.array = array

    def __arrow_c_array__(self, requested_schema=None):
        return self.array.__arrow_c_array__(requested_schema=requested_schema)


class StreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def __arrow_c_stream__(self, requested_schema=None):
        return self.stream.__arrow_c_stream__(requested_schema=requested_schema)


def test_schema_import():
    pa_schema = pa.schema([pa.field("some_name", pa.int32())])

    for schema_obj in [pa_schema, SchemaWrapper(pa_schema)]:
        schema = na.schema(schema_obj)
        # some basic validation
        assert schema.is_valid()
        assert schema.format == "+s"
        assert schema._to_string(recursive=True) == "struct<some_name: int32>"


def test_array_import():
    pa_arr = pa.array([1, 2, 3], pa.int32())

    for arr_obj in [pa_arr, ArrayWrapper(pa_arr)]:
        array = na.array(arr_obj)
        # some basic validation
        assert array.is_valid()
        assert array.length == 3
        assert array.schema._to_string(recursive=True) == "int32"


def test_array_stream_import():
    def make_reader():
        pa_array_child = pa.array([1, 2, 3], pa.int32())
        pa_array = pa.record_batch([pa_array_child], names=["some_column"])
        return pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])

    for stream_obj in [make_reader(), StreamWrapper(make_reader())]:
        array_stream = na.array_stream(stream_obj)
        # some basic validation
        assert array_stream.is_valid()
        array = array_stream.get_next()
        assert array.length == 3
        assert (
            array_stream.get_schema()._to_string(recursive=True)
            == "struct<some_column: int32>"
        )
