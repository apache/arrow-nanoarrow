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

project(
    'nanoarrow',
    'c', 'cpp',
    version: '0.5.0-SNAPSHOT',
    license: 'Apache 2.0',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c99',
        'warning_level=2',
        'cpp_std=c++17',
    ]
)

if get_option('NANOARROW_CODE_COVERAGE')
  error(
      '''For meson builds, please use '-Db_coverage=true' flag
      instead of -DNANOARROW_CODE_COVERAGE=true''')
endif

if get_option('NANOARROW_BUNDLE')
  error('NANOARROW_BUNDLE not implemented in meson configuration - try CMake instead')
endif

if get_option('NANOARROW_BUNDLE_AS_CPP')
  error('NANOARROW_BUNDLE_AS_CPP not implemented in meson configuration - try CMake instead')
endif

subdir('src/nanoarrow')

if get_option('NANOARROW_BUILD_BENCHMARKS')
  subdir('dev/benchmarks')
endif
