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


def make_class_label(obj, module=None):
    if module is None:
        module = obj.__class__.__module__
    return f"{module}.{obj.__class__.__name__}"


def c_schema_to_string(obj, max_char_width=80):
    max_char_width = max(max_char_width, 10)
    c_schema_string = obj._to_string(recursive=True, max_chars=max_char_width + 1)
    if len(c_schema_string) > max_char_width:
        return c_schema_string[: (max_char_width - 3)] + "..."
    else:
        return c_schema_string


def schema_repr(schema, indent=0):
    indent_str = " " * indent
    class_label = make_class_label(schema, module="nanoarrow.c_lib")
    if schema._addr() == 0:
        return f"<{class_label} <NULL>>"
    elif not schema.is_valid():
        return f"<{class_label} <released>>"

    lines = [f"<{class_label} {schema._to_string()}>"]

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


def array_repr(array, indent=0, max_char_width=80):
    if max_char_width < 20:
        max_char_width = 20

    indent_str = " " * indent
    class_label = make_class_label(array, module="nanoarrow.c_lib")
    if array._addr() == 0:
        return f"<{class_label} <NULL>>"
    elif not array.is_valid():
        return f"<{class_label} <released>>"

    schema_string = array.schema._to_string(
        max_chars=max_char_width - indent - 23, recursive=True
    )
    lines = [f"<{class_label} {schema_string}>"]
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


def schema_view_repr(schema_view):
    class_label = make_class_label(schema_view, module="nanoarrow.c_lib")

    lines = [
        f"<{class_label}>",
        f"- type: {repr(schema_view.type)}",
        f"- storage_type: {repr(schema_view.storage_type)}",
    ]

    for attr_name in sorted(dir(schema_view)):
        if attr_name.startswith("_") or attr_name in ("type", "storage_type"):
            continue

        attr_value = getattr(schema_view, attr_name)
        if attr_value is None:
            continue

        lines.append(f"- {attr_name}: {repr(attr_value)}")

    return "\n".join(lines)


def array_view_repr(array_view, max_char_width=80, indent=0):
    indent_str = " " * indent
    class_label = make_class_label(array_view, module="nanoarrow.c_lib")

    lines = [f"<{class_label}>"]

    for attr in ("storage_type", "length", "offset", "null_count"):
        attr_repr = repr(getattr(array_view, attr))
        lines.append(f"{indent_str}- {attr}: {attr_repr}")

    lines.append(f"{indent_str}- buffers[{array_view.n_buffers}]:")
    for i, buffer in enumerate(array_view.buffers):
        buffer_type = array_view.buffer_type(i)
        lines.append(
            f"{indent_str}  - {buffer_type} "
            f"<{buffer_view_repr(buffer, max_char_width - indent - 4 - len(buffer))}>"
        )

    if array_view.dictionary:
        dictionary_repr = array_view_repr(
            array_view.dictionary, max_char_width=max_char_width, indent=indent + 2
        )
        lines.append(f"{indent_str}- dictionary: {dictionary_repr}")
    else:
        lines.append(f"{indent_str}- dictionary: NULL")

    lines.append(f"{indent_str}- children[{array_view.n_children}]:")
    for child in array_view.children:
        child_repr = array_view_repr(
            child, max_char_width=max_char_width, indent=indent + 4
        )
        lines.append(f"{indent_str}  - {child_repr}")

    return "\n".join(lines)


def buffer_view_repr(buffer_view, max_char_width=80):
    if max_char_width < 20:
        max_char_width = 20

    prefix = f"{buffer_view.data_type}"
    prefix += f"[{buffer_view.size_bytes} b]"

    if buffer_view.device.device_type_id == 1:
        return (
            prefix
            + " "
            + buffer_view_preview_cpu(buffer_view, max_char_width - len(prefix) - 2)
        )
    else:
        return prefix


def buffer_view_preview_cpu(buffer_view, max_char_width):
    if buffer_view.element_size_bits == 0:
        preview_elements = max_char_width - 3
        joined = repr(bytes(memoryview(buffer_view)[:preview_elements]))
    elif buffer_view.element_size_bits == 1:
        max_elements = max_char_width // 8
        if max_elements > len(buffer_view):
            preview_elements = len(buffer_view)
        else:
            preview_elements = max_elements

        joined = "".join(
            "".join(reversed(format(buffer_view[i], "08b")))
            for i in range(preview_elements)
        )
    else:
        max_elements = max_char_width // 3
        if max_elements > len(buffer_view):
            preview_elements = len(buffer_view)
        else:
            preview_elements = max_elements

        joined = " ".join(repr(buffer_view[i]) for i in range(preview_elements))

    if len(joined) > max_char_width or preview_elements < len(buffer_view):
        return joined[: (max_char_width - 3)] + "..."
    else:
        return joined


def array_stream_repr(array_stream, max_char_width=80):
    class_label = make_class_label(array_stream, module="nanoarrow.c_lib")

    if array_stream._addr() == 0:
        return f"<{class_label} <NULL>>"
    elif not array_stream.is_valid():
        return f"<{class_label} <released>>"

    lines = [f"<{class_label}>"]
    try:
        schema = array_stream.get_schema()
        schema_string = schema._to_string(max_chars=max_char_width - 16, recursive=True)
        lines.append(f"- get_schema(): {schema_string}")
    except Exception as e:
        lines.append(f"- get_schema(): <error calling get_schema(): {e}>")

    return "\n".join(lines)


def device_array_repr(device_array):
    class_label = make_class_label(device_array, module="nanoarrow.device")

    title_line = f"<{class_label}>"
    device_type = (
        f"- device_type: {device_array.device_type.name} "
        f"<{device_array.device_type_id}>"
    )
    device_id = f"- device_id: {device_array.device_id}"
    array = f"- array: {array_repr(device_array.array, indent=2)}"
    return "\n".join((title_line, device_type, device_id, array))


def device_repr(device):
    class_label = make_class_label(device, module="nanoarrow.device")

    title_line = f"<{class_label}>"
    device_type = f"- device_type: {device.device_type.name} <{device.device_type_id}>"
    device_id = f"- device_id: {device.device_id}"
    return "\n".join([title_line, device_type, device_id])
