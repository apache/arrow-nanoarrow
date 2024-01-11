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

from nanoarrow._lib import c_version  # noqa: F401
from nanoarrow.c_lib import (  # noqa: F401
    c_schema,
    c_array,
    c_array_stream,
    c_schema_view,
    c_array_view,
    allocate_c_schema,
    allocate_c_array,
    allocate_c_array_stream,
)
from nanoarrow._version import __version__  # noqa: F401
