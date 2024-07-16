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

# cython: language_level = 3

from nanoarrow_c cimport (
    ArrowLayout,
    ArrowMetadataReader,
    ArrowSchema,
    ArrowSchemaView
)

cpdef assert_type_equal(object lhs, object rhs, bint check_nullability)

cdef class CLayout:
    cdef ArrowLayout* _layout
    cdef object _base
    cdef int _n_buffers


cdef class SchemaMetadata:
    cdef object _base
    cdef const char* _metadata
    cdef ArrowMetadataReader _reader

    cdef _init_reader(self)


cdef class CSchema:
    # Currently, _base is always the capsule holding the root of a tree of ArrowSchemas
    # (but in general is just a strong reference to an object whose Python lifetime is
    # used to guarantee that _ptr is valid).
    cdef object _base
    cdef ArrowSchema* _ptr


cdef class CSchemaView:
    # _base is currently only a CSchema (but in general is just an object whose Python
    # lifetime guarantees that the pointed-to data from ArrowStringViews remains valid
    cdef object _base
    cdef ArrowSchemaView _schema_view
    # Not part of the ArrowSchemaView (but possibly should be)
    cdef bint _dictionary_ordered
    cdef bint _nullable
    cdef bint _map_keys_sorted


cdef class CSchemaBuilder:
    cdef CSchema c_schema
    cdef ArrowSchema* _ptr
