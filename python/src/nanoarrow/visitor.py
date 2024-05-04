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

from typing import Any, Callable, Iterable, List, Mapping, Sequence, Union

from nanoarrow._lib import CArrayView, CArrowType, CBuffer
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.c_buffer import CBufferBuilder
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator


def to_columns(obj, schema=None) -> Mapping[str, Sequence]:
    return ColumnsBuilder.visit(obj, schema)


def to_pylist(obj, schema=None) -> List:
    return ListBuilder.visit(obj, schema)


class ArrayStreamVisitor(ArrayViewBaseIterator):
    @classmethod
    def visit(cls, obj, schema=None, **kwargs):
        if hasattr(obj, "__len__"):
            total_elements = len(obj)
        else:
            total_elements = None

        with c_array_stream(obj, schema=schema) as stream:
            visitor = cls(stream._get_cached_schema(), **kwargs)
            state = visitor.begin(total_elements)

            iterator_set_array = visitor._set_array
            visit_chunk_view = visitor.visit_chunk_view
            array_view = visitor._array_view

            for array in stream:
                iterator_set_array(array)
                state = visit_chunk_view(array_view, state)

        return visitor.finish(state)

    def begin(self, total_elements: Union[int, None] = None):
        return None

    def visit_chunk_view(self, array_view: CArrayView, state: Any) -> Any:
        return state

    def finish(self, state: Any) -> Any:
        return state


class ColumnsBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)

        self._child_visitors = []
        self._child_array_views = []
        for child_schema, child_array_view in zip(
            self._schema.children, self._array_view.children
        ):
            self._child_visitors.append(
                self._resolve_child_visitor(child_schema, child_array_view)
            )
            self._child_array_views.append(child_array_view)

    def _resolve_child_visitor(self, child_schema, child_array_view):
        return ListBuilder(child_schema, _array_view=child_array_view)

    def begin(self, total_elements: Union[int, None] = None):
        child_visitors = self._child_visitors
        child_state = [child.begin(total_elements) for child in child_visitors]
        return child_visitors, child_state, self._child_array_views

    def visit_chunk_view(self, array_view: CArrayView, state: Any) -> Any:
        child_visitors, child_state, child_array_views = state

        for i in range(len(child_visitors)):
            child_state[i] = child_visitors[i].visit_chunk_view(
                child_array_views[i], child_state[i]
            )

        return state

    def finish(self, state: Any) -> Any:
        child_visitors, child_state, _ = state

        out = {}
        for i, schema in enumerate(self._schema.children):
            out[schema.name] = child_visitors[i].finish(child_state[i])

        return out


class ListBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, iterator_cls=PyIterator, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        self._iterator = iterator_cls(schema, _array_view=self._array_view)

    def begin(self, total_elements: Union[int, None] = None):
        return self._iterator._iter_chunk, []

    def visit_chunk_view(self, array_view: CArrayView, state: Any):
        iter_chunk, out = state
        out.extend(iter_chunk(0, array_view.length))
        return iter_chunk, out

    def finish(self, state: Any):
        return state[1]


class BufferConcatenator(ArrayStreamVisitor):
    def __init__(self, schema, *, buffer_index=1, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        self._buffer_index = buffer_index

    def begin(self, total_elements: Union[int, None] = None):
        buffer_index = self._buffer_index
        buffer_data_type = self._schema_view.layout.buffer_data_type_id[buffer_index]
        element_size_bits = self._schema_view.layout.element_size_bits[buffer_index]
        element_size_bytes = element_size_bits // 8

        builder = CBufferBuilder()
        builder.set_data_type(buffer_data_type)

        if total_elements is not None:
            builder.reserve_bytes(total_elements * element_size_bytes)

        return 0, buffer_index, builder

    def visit_chunk_view(self, array_view: CArrayView, state: Any) -> Any:
        out_start, buffer_index, writable_buffer = state
        offset = array_view.offset
        length = array_view.length

        src = memoryview(array_view.buffer(buffer_index))[offset : offset + length]
        dst_bytes = len(src) * src.itemsize
        writable_buffer.reserve_bytes(dst_bytes)

        dst = memoryview(writable_buffer).cast(src.format)
        dst[out_start : out_start + length] = src
        writable_buffer.advance(dst_bytes)

        return out_start + length, buffer_index, writable_buffer

    def finish(self, state: Any) -> CBuffer:
        return state[2].finish()


class UnpackedBitmapConcatenator(BufferConcatenator):
    def begin(self, iterator: ArrayViewBaseIterator):
        buffer_index = self._buffer
        builder = CBufferBuilder()
        builder.set_data_type(CArrowType.UINT8)

        if self._total_elements is not None:
            builder.reserve_bytes(self._total_elements)

        return 0, buffer_index, builder

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        out_start, buffer_index, writable_buffer = state
        offset = array_view.offset
        length = array_view.length

        writable_buffer.reserve_bytes(length)
        array_view.buffer(buffer_index).unpack_bits_into(
            writable_buffer, offset, length, out_start
        )
        writable_buffer.advance(length)

        return out_start + length, buffer_index, writable_buffer


class NullableBufferConcatenator(UnpackedBitmapConcatenator):
    def __init__(self, *, parent=None, total_elements=None) -> None:
        super().__init__(buffer=0, total_elements=total_elements)
        self._parent = parent

    def begin(self, iterator: ArrayViewBaseIterator):
        return self._parent, self._parent.begin(iterator), super().begin(iterator)

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        parent, parent_state, super_state = state
        parent_state = parent.visit_array(array_view, iterator, parent_state)
        super_state = super().visit_array(array_view, iterator, state)
        return parent, parent_state, super_state

    def finish(self, state):
        parent, parent_state, super_state = state
        return parent.finish(parent_state), super().finish(super_state)
