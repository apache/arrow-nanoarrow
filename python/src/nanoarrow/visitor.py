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

from typing import Any, Callable, Iterable

from nanoarrow._lib import CArrayView, CArrowType
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.c_buffer import CBufferBuilder
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator


class ArrayStreamVisitor:
    @classmethod
    def visit(cls, obj, schema=None, **kwargs):
        visitor = cls(**kwargs)

        with c_array_stream(obj, schema=schema) as stream:
            iterator = visitor._iterator_cls(stream._get_cached_schema())
            state = visitor.begin(iterator)

            iterator_set_array = iterator._set_array
            iterator_iter_chunk = iterator._iter_chunk
            visit_array = visitor.visit_array
            array_view = iterator._array_view

            for array in stream:
                iterator_set_array(array)
                state = visit_array(array_view, iterator_iter_chunk, state)

        return visitor.finish(state)

    def __init__(self, *, iterator_cls=ArrayViewBaseIterator) -> None:
        self._iterator_cls = iterator_cls

    def begin(self, iterator: ArrayViewBaseIterator):
        return None

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        return state

    def finish(self, state):
        return state


class ColumnsBuilder(ArrayStreamVisitor):

    def __init__(self, *, total_elements=None) -> None:
        super().__init__(iterator_cls=PyIterator)

        self._total_elements = total_elements

    def _resolve_visitor(self, schema_view, nullable=None):
        if nullable is None:
            nullable = schema_view.nullable

        if schema_view.buffer_format is not None and not nullable:
            return BufferConcatenator(buffer=1, total_elements=self._total_elements)

        elif schema_view.type_id == CArrowType.BOOL and not nullable:
            return UnpackedBitmapConcatenator(
                buffer=1, total_elements=self._total_elements
            )

        elif (
            schema_view.buffer_format is not None
            or schema_view.type_id == CArrowType.BOOL
        ):
            return NullableBufferConcatenator(
                parent=self._resolve_visitor(schema_view, nullable=False),
                total_elements=self._total_elements,
            )

        else:
            return ListBuilder()

    def _handle_nullable_buffer(self, mask, buffer):
        return mask, buffer

    def begin(self, iterator: ArrayViewBaseIterator):
        if iterator._schema_view.type != "struct":
            raise ValueError(
                f"Can't build columns from type '{iterator._schema_view.type}'"
            )

        state = []
        for child_iterator in iterator._children:
            child = self._resolve_visitor(child_iterator._schema_view)
            state.append((child, child.begin(child_iterator), iterator._schema.name))

        return state

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        for i in range(len(state)):
            child, child_state, name = state[i]
            child_state = child.visit_array(array_view.child(i), None, child_state)
            state[i] = child, child_state, name

    def finish(self, state):
        out = {}
        for i in range(len(state)):
            child, child_state, name = state[i]
            child_state = child.finish(child_state)
            if isinstance(child, NullableBufferConcatenator):
                child_state = self._handle_nullable_buffer(*child_state)

            out[name] = child_state

        return out


class ListBuilder(ArrayStreamVisitor):
    def __init__(self, *, iterator_cls=PyIterator, total_elements=None) -> None:
        super().__init__(iterator_cls=iterator_cls)

    def begin(self, iterator: ArrayViewBaseIterator):
        return []

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        state.extend(iterator(0, array_view.length))
        return state


class BufferConcatenator(ArrayStreamVisitor):
    def __init__(self, *, buffer=1, total_elements=None) -> None:
        super().__init__(iterator_cls=ArrayViewBaseIterator)
        self._buffer = buffer
        self._total_elements = total_elements

    def begin(self, iterator: ArrayViewBaseIterator):
        buffer_index = self._buffer
        buffer_data_type = iterator._schema_view.layout.buffer_data_type_id[
            buffer_index
        ]
        element_size_bits = iterator._schema_view.layout.element_size_bits[buffer_index]
        element_size_bytes = element_size_bits // 8

        builder = CBufferBuilder()
        builder.set_data_type(buffer_data_type)

        if self._total_elements is not None:
            builder.reserve_bytes(self._total_elements * element_size_bytes)

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

        src = memoryview(array_view.buffer(buffer_index))[offset : offset + length]
        dst_bytes = len(src) * src.itemsize
        writable_buffer.reserve_bytes(dst_bytes)

        dst = memoryview(writable_buffer).cast(src.format)
        dst[out_start : out_start + length] = src
        writable_buffer.advance(dst_bytes)

        return out_start + length, buffer_index, writable_buffer

    def finish(self, state):
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
