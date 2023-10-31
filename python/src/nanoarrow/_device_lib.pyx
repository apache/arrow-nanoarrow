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


from libc.stdint cimport uintptr_t
from nanoarrow_device_c cimport ArrowDeviceArray


cdef class DeviceArrayHolder:
    """Memory holder for an ArrowDeviceArray

    This class is responsible for the lifecycle of the ArrowArray
    whose memory it is responsible. When this object is deleted,
    a non-NULL release callback is invoked.
    """
    cdef ArrowDeviceArray c_array

    def __cinit__(self):
        self.c_array.array.release = NULL

    def __dealloc__(self):
        if self.c_array.array.release != NULL:
          self.c_array.array.release(&self.c_array.array)

    def _addr(self):
        return <uintptr_t>&self.c_array
