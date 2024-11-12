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

project('flatcc', 'c', version: '0.6.1', license: 'Apache-2.0')

incdir = include_directories('include')

flatcc_lib = library(
    'flatcc',
    sources: [
        'src/runtime/builder.c',
        'src/runtime/emitter.c',
        'src/runtime/verifier.c',
        'src/runtime/refmap.c',
    ],
    include_directories: [incdir],
)

flatcc_dep = declare_dependency(
    include_directories: [incdir],
    link_with: [flatcc_lib],
)
