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

# The functions here are imported in _lib.pyx. They're defined here
# instead of there to make it easier to iterate (no need to rebuild
# after editing when working with an editable installation)


def schema_repr(schema, indent=0):
    indent_str = " " * indent
    if schema._addr() == 0:
        return "<NULL nanoarrow.clib.CSchema>"
    elif not schema.is_valid():
        return "<released nanoarrow.clib.CSchema>"

    lines = [f"<nanoarrow.clib.CSchema {schema._to_string()}>"]

    for attr in ("format", "name", "flags"):
        attr_repr = repr(getattr(schema, attr))
        lines.append(f"{indent_str}- {attr}: {attr_repr}")

    metadata = schema.metadata
    if schema.metadata is None:
        lines.append(f"{indent_str}- metadata: NULL")
    else:
        lines.append(f"{indent_str}- metadata:")
        for key, value in metadata:
            lines.append(f"{indent_str}  - {repr(key)}: {repr(value)}")

    if schema.dictionary:
        dictionary_repr = schema_repr(schema.dictionary, indent=indent + 2)
        lines.append(f"{indent_str}- dictionary: {dictionary_repr}")
    else:
        lines.append(f"{indent_str}- dictionary: NULL")

    lines.append(f"{indent_str}- children[{schema.n_children}]:")
    for child in schema.children:
        child_repr = schema_repr(child, indent=indent + 4)
        lines.append(f"{indent_str}  {repr(child.name)}: {child_repr}")

    return "\n".join(lines)


def array_repr(array, indent=0):
    indent_str = " " * indent
    if array._addr() == 0:
        return "<NULL nanoarrow.clib.CArray>"
    elif not array.is_valid():
        return "<released nanoarrow.clib.CArray>"

    lines = [f"<nanoarrow.clib.CArray {array.schema._to_string()}>"]
    for attr in ("length", "offset", "null_count", "buffers"):
        attr_repr = repr(getattr(array, attr))
        lines.append(f"{indent_str}- {attr}: {attr_repr}")

    if array.dictionary:
        dictionary_repr = array_repr(array.dictionary, indent=indent + 2)
        lines.append(f"{indent_str}- dictionary: {dictionary_repr}")
    else:
        lines.append(f"{indent_str}- dictionary: NULL")

    lines.append(f"{indent_str}- children[{array.n_children}]:")
    for child in array.children:
        child_repr = array_repr(child, indent=indent + 4)
        lines.append(f"{indent_str}  {repr(child.schema.name)}: {child_repr}")

    return "\n".join(lines)


def device_array_repr(device_array):
    title_line = "<nanoarrow.device.clib.CDeviceArray>"
    device_type = f"- device_type: {device_array.device_type}"
    device_id = f"- device_id: {device_array.device_id}"
    array = f"- array: {array_repr(device_array.array, indent=2)}"
    return "\n".join((title_line, device_type, device_id, array))


def device_repr(device):
    title_line = "<nanoarrow.device.Device>"
    device_type = f"- device_type: {device.device_type}"
    device_id = f"- device_id: {device.device_id}"
    return "\n".join([title_line, device_type, device_id])
