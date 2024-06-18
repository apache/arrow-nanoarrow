# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

fetchcontent_declare(nanoarrow_zlib
                     URL https://github.com/madler/zlib/releases/download/v1.3/zlib13.zip
                     URL_HASH SHA256=c561d09347f674f0d72692e7c75d9898919326c532aab7f8c07bb43b07efeb38
)
fetchcontent_makeavailable(nanoarrow_zlib)

add_library(ZLIB::ZLIB ALIAS zlibstatic)
target_include_directories(zlibstatic INTERFACE ${zlib_BINARY_DIR} ${zlib_SOURCE_DIR})
