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

from libc.stdint cimport int32_t, int64_t, uintptr_t

from nanoarrow_c cimport ArrowArray

cdef extern from "nanoarrow_device.h" nogil:

    ctypedef int32_t ArrowDeviceType

    struct ArrowDeviceArray:
        ArrowArray array
        int64_t device_id
        ArrowDeviceType device_type
        void* sync_event
