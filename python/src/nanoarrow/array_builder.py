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

from nanoarrow._lib import CArrayBuilder, CArrowType, CBuffer, CSchemaBuilder


def c_array_from_pybuffer(obj):
    buffer = CBuffer().set_pybuffer(obj)
    view = buffer.data
    type_id = view.data_type_id
    element_size_bits = view.element_size_bits

    array_builder = CArrayBuilder.allocate()

    # Fixed-size binary needs a schema
    if type_id == CArrowType.BINARY and element_size_bits != 0:
        c_schema = (
            CSchemaBuilder.allocate()
            .set_type_fixed_size(CArrowType.FIXED_SIZE_BINARY, element_size_bits // 8)
            .finish()
        )
        array_builder.init_from_schema(c_schema)
    elif type_id == CArrowType.STRING:
        array_builder.init_from_type(int(CArrowType.INT8))
    elif type_id == CArrowType.BINARY:
        array_builder.init_from_type(int(CArrowType.UINT8))
    else:
        array_builder.init_from_type(int(type_id))

    # Set the length
    array_builder.set_length(len(view))

    # Move ownership of the ArrowBuffer wrapped by buffer to array_builder.buffer(1)
    array_builder.set_buffer(1, buffer)

    # No nulls or offset from a PyBuffer
    array_builder.set_null_count(0)
    array_builder.set_offset(0)

    return array_builder.finish()
