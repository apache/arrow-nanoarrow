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

py = import('python').find_installation(pure: false)

cython_args = [
    '--include-dir',
    meson.current_build_dir(),
    '--include-dir',
    meson.current_source_dir(), # TODO: should meson handle this natively?
]
if get_option('buildtype') == 'debug'
    cython_args += ['--gdb']
endif

# TODO: not sure these are the idiomatic approach to handle multiple dependencies
# from same project, where some require optionas being set
nanoarrow_dep = dependency('nanoarrow', default_options: ['ipc=true'])
#nanoarrow_device_dep = dependency('nanoarrow_device', default_options: ['device=true'])
nanoarrow_ipc_dep = dependency('nanoarrow_ipc', default_options: ['ipc=true'])

generated_pyx = custom_target(
    'generate-pyx',
    input: meson.current_build_dir(),
    output: 'nanoarrow_c.pxd',
    command: [py, meson.current_source_dir() + '/../../bootstrap.py', '@INPUT@'],
)
nanoarrow_c_dep = declare_dependency(sources: generated_pyx)

py.extension_module(
    '_lib',
    sources: ['_lib.pyx'],
    cython_args: cython_args,
    dependencies: [
        nanoarrow_c_dep,
        nanoarrow_dep,
    ],
    subdir: 'nanoarrow/',
    install: true,
)

py.extension_module(
    '_ipc_lib',
    sources: ['_ipc_lib.pyx'],
    cython_args: cython_args,
    dependencies: [
        nanoarrow_c_dep,
        nanoarrow_dep,
        nanoarrow_ipc_dep,
    ],
    subdir: 'nanoarrow',
    install: true,
)

py_sources = [
    'array.py',
    'array_stream.py',
    'c_array.py',
    'c_array_stream.py',
    'c_buffer.py',
    'c_schema.py',
    'device.py',
    'ipc.py',
    'iterator.py',
    '_repr_utils.py',
    'schema.py',
    '_static_version.py',
    '_version.py',
    'visitor.py',
]

foreach source: py_sources
    py.install_sources(
        source,
        subdir: 'nanoarrow'
    )
endforeach
