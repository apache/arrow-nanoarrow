<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

<!-- Render with jupyter nbconvert --to markdown README.ipynb -->

# nanoarrow for Python

The nanoarrow Python package provides bindings to the nanoarrow C library. Like
the nanoarrow C library, it provides tools to facilitate the use of the
[Arrow C Data](https://arrow.apache.org/docs/format/CDataInterface.html)
and [Arrow C Stream](https://arrow.apache.org/docs/format/CStreamInterface.html)
interfaces.

## Installation

Python bindings for nanoarrow are not yet available on PyPI. You can install via
URL (requires a C compiler):

```bash
python -m pip install "git+https://github.com/apache/arrow-nanoarrow.git#egg=nanoarrow&subdirectory=python"
```

If you can import the namespace, you're good to go!


```python
import nanoarrow as na
```

## Low-level C library bindings

The Arrow C Data and Arrow C Stream interfaces are comprised of three structures: the `ArrowSchema` which represents a data type of an array, the `ArrowArray` which represents the values of an array, and an `ArrowArrayStream`, which represents zero or more `ArrowArray`s with a common `ArrowSchema`.

### Schemas

Use `nanoarrow.c_schema()` to convert an object to an `ArrowSchema` and wrap it as a Python object. This works for any object implementing the [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface.html) (e.g., `pyarrow.Schema`, `pyarrow.DataType`, and `pyarrow.Field`).


```python
import pyarrow as pa
schema = na.c_schema(pa.decimal128(10, 3))
schema
```




    <nanoarrow.c_lib.CSchema decimal128(10, 3)>
    - format: 'd:10,3'
    - name: ''
    - flags: 2
    - metadata: NULL
    - dictionary: NULL
    - children[0]:



You can extract the fields of a `CSchema` object one at a time or parse it into a view to extract deserialized parameters.


```python
na.c_schema_view(schema)
```




    <nanoarrow.c_lib.CSchemaView>
    - type: 'decimal128'
    - storage_type: 'decimal128'
    - decimal_bitwidth: 128
    - decimal_precision: 10
    - decimal_scale: 3
    - dictionary_ordered: False
    - map_keys_sorted: False
    - nullable: True
    - storage_type_id: 24
    - type_id: 24



Advanced users can allocate an empty `CSchema` and populate its contents by passing its `._addr()` to a schema-exporting function.


```python
schema = na.allocate_c_schema()
pa.int32()._export_to_c(schema._addr())
schema
```




    <nanoarrow.c_lib.CSchema int32>
    - format: 'i'
    - name: ''
    - flags: 2
    - metadata: NULL
    - dictionary: NULL
    - children[0]:



The `CSchema` object cleans up after itself: when the object is deleted, the underlying `ArrowSchema` is released.

### Arrays

You can use `nanoarrow.c_array()` to convert an array-like object to an `ArrowArray`, wrap it as a Python object, and attach a schema that can be used to interpret its contents. This works for any object implementing the [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface.html) (e.g., `pyarrow.Array`, `pyarrow.RecordBatch`).


```python
array = na.c_array(pa.array(["one", "two", "three", None]))
array
```




    <nanoarrow.c_lib.CArray string>
    - length: 4
    - offset: 0
    - null_count: 1
    - buffers: (3678035706048, 3678035705984, 3678035706112)
    - dictionary: NULL
    - children[0]:



You can extract the fields of a `CArray` one at a time or parse it into a view to extract deserialized content:


```python
na.c_array_view(array)
```




    <nanoarrow.c_lib.CArrayView>
    - storage_type: 'string'
    - length: 4
    - offset: 0
    - null_count: 1
    - buffers[3]:
      - validity <bool[1 b] 11100000>
      - data_offset <int32[20 b] 0 3 6 11 11>
      - data <string[11 b] b'onetwothree'>
    - dictionary: NULL
    - children[0]:



Like the `CSchema`, you can allocate an empty one and access its address with `_addr()` to pass to other array-exporting functions.


```python
array = na.allocate_c_array()
pa.array([1, 2, 3])._export_to_c(array._addr(), array.schema._addr())
array.length
```




    3



### Array streams

You can use `nanoarrow.c_array_stream()` to wrap an object representing a sequence of `CArray`s with a common `CSchema` to an `ArrowArrayStream` and wrap it as a Python object. This works for any object implementing the [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface.html) (e.g., `pyarrow.RecordBatchReader`).


```python
pa_array_child = pa.array([1, 2, 3], pa.int32())
pa_array = pa.record_batch([pa_array_child], names=["some_column"])
reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
array_stream = na.c_array_stream(reader)
array_stream
```




    <nanoarrow.c_lib.CArrayStream>
    - get_schema(): struct<some_column: int32>



You can pull the next array from the stream using `.get_next()` or use it like an iterator. The `.get_next()` method will raise `StopIteration` when there are no more arrays in the stream.


```python
for array in array_stream:
    print(array)
```

    <nanoarrow.c_lib.CArray struct<some_column: int32>>
    - length: 3
    - offset: 0
    - null_count: 0
    - buffers: (0,)
    - dictionary: NULL
    - children[1]:
      'some_column': <nanoarrow.c_lib.CArray int32>
        - length: 3
        - offset: 0
        - null_count: 0
        - buffers: (0, 3678035837056)
        - dictionary: NULL
        - children[0]:


You can also get the address of a freshly-allocated stream to pass to a suitable exporting function:


```python
array_stream = na.allocate_c_array_stream()
reader._export_to_c(array_stream._addr())
array_stream
```




    <nanoarrow.c_lib.CArrayStream>
    - get_schema(): struct<some_column: int32>



## Development

Python bindings for nanoarrow are managed with [setuptools](https://setuptools.pypa.io/en/latest/index.html).
This means you can build the project using:

```shell
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/python
pip install -e .
```

Tests use [pytest](https://docs.pytest.org/):

```shell
# Install dependencies
pip install -e .[test]

# Run tests
pytest -vvx
```
