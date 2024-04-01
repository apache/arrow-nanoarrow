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

import nanoarrow as na


class SchemaSuite:
    """
    Benchmarks of some Schema/CSchema operations
    """

    def setup(self):
        self.children = [na.int32()] * 10000
        self.c_children = [na.c_schema(child) for child in self.children]
        self.c_wide_struct = na.c_schema(na.struct(self.children))

    def time_create_wide_struct_from_schemas(self):
        """Create a struct Schema with 10000 columns from a list of Schema"""
        na.struct(self.children)

    def time_create_wide_struct_from_c_schemas(self):
        """Create a struct Schema with 10000 columns from a list of CSchema"""
        na.struct(self.c_children)

    def time_c_schema_protocol_wide_struct(self):
        """Export a struct Schema with 10000 columns via the PyCapsule protocol"""
        self.c_wide_struct.__arrow_c_schema__()
