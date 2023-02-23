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

from libc.stdint cimport int64_t

cdef extern from "nanoarrow.h":
    cdef int ARROW_FLAG_DICTIONARY_ORDERED
    cdef int ARROW_FLAG_NULLABLE
    cdef int ARROW_FLAG_MAP_KEYS_SORTED

    cdef struct ArrowSchema:
        const char* format
        const char* name
        const char* metadata
        int64_t flags
        int64_t n_children
        ArrowSchema** children
        ArrowSchema* dictionary
        void (*release)(ArrowSchema*)
        void* private_data

    cdef struct ArrowArray:
        int64_t length
        int64_t null_count
        int64_t offset
        int64_t n_buffers
        int64_t n_children
        const void** buffers
        ArrowArray** children
        ArrowArray* dictionary
        void (*release)(ArrowArray*)
        void* private_data

    cdef struct ArrowArrayStream:
        int (*get_schema)(ArrowArrayStream* stream, ArrowSchema* out)
        int (*get_next)(ArrowArrayStream* stream, ArrowArray* out)
        const char* (*get_last_error)(ArrowArrayStream*)
        void (*release)(ArrowArrayStream* stream)
        void* private_data
