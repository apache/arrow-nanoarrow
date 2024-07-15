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

include(FetchContent)

fetchcontent_declare(nlohmann_json
                     URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
                     URL_HASH SHA256=04022b05d806eb5ff73023c280b68697d12b93e1b7267a0b22a1a39ec7578069
)
fetchcontent_makeavailable(nlohmann_json)
