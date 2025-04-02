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

project(
    'nanoarrow',
    'c',
    'cython',
    version: run_command(['src/nanoarrow/_version.py', '--print'], check: true).stdout().strip(),
    license: 'Apache-2.0',
    meson_version: '>=1.2.0',
    default_options: [
        'warning_level=2',
        'c_std=c99',
        'default_library=static',
        'force_fallback_for=zstd',
        # We need to set these options at the project default_option level
        # due to https://github.com/mesonbuild/meson/issues/6728
        'arrow-nanoarrow:ipc=enabled',
        'arrow-nanoarrow:ipc_with_zstd=enabled',
        'arrow-nanoarrow:device=enabled',
        'arrow-nanoarrow:namespace=PythonPkg',
        'zstd:bin_programs=false',
    ],
)

subdir('src/nanoarrow')

meson.add_dist_script('python', meson.current_source_dir() / 'generate_dist.py')
