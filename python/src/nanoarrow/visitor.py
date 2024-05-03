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

    def _copy_method_bitmap(self, buffer_view, builder, offset, length, dest_offset):
        buffer_view.unpack_bits_into(builder, offset, length, dest_offset)
        builder.advance(length)

    def _copy_method_non_bitmap(
        self, buffer_view, builder, offset, length, dest_offset
    ):
        src = memoryview(buffer_view)[offset : offset + length]
        dst = memoryview(builder).cast(src.format)
        dst[dest_offset : dest_offset + length] = src
        builder.advance(len(src))

    def begin(self, iterator: ArrayViewBaseIterator):
        buffer_index = self._buffer
        buffer_data_type = iterator._schema_view.layout.buffer_data_type_id[
            buffer_index
        ]
        element_size_bits = iterator._schema_view.layout.element_size_bits[buffer_index]

        if buffer_data_type == CArrowType.BOOL:
            element_size_bytes = 1
            buffer_data_type = CArrowType.UINT8
            copy_method = self._copy_method_bitmap
        else:
            element_size_bytes = element_size_bits // 8
            copy_method = self._copy_method_non_bitmap

        builder = CBufferBuilder()
        builder.set_data_type(buffer_data_type)

        if self._total_elements is not None:
            builder.reserve_bytes(self._total_elements * element_size_bytes)

        return 0, buffer_index, builder, copy_method

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        out_start, buffer_index, writable_buffer, copy_method = state
        offset = array_view.offset
        length = array_view.length
        copy_method(
            array_view.buffer(buffer_index), writable_buffer, offset, length, out_start
        )

        return out_start + length, buffer_index, writable_buffer, copy_method

    def finish(self, state):
        return state[2].finish()


class NumpyObjectArrayBuilder(ArrayStreamVisitor):
    def __init__(self, *, iterator_cls=PyIterator, total_elements=None) -> None:
        super().__init__(iterator_cls=iterator_cls)
        self._total_elements = total_elements

    def begin(self, iterator: ArrayViewBaseIterator):
        from numpy import empty, fromiter

        array = empty(self._total_elements, "O")
        return 0, array, array.dtype, fromiter

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        start, array, dtype, fromiter = state
        length = array_view.length
        end = start + length
        array[start:end] = fromiter(iterator(0, length), dtype, length)

        return end, array, dtype, fromiter

    def finish(self, state):
        return state[1]
