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

flatcc_dep = dependency('flatcc')
zstd_dep = dependency('libzstd')
nanoarrow_proj = subproject('arrow-nanoarrow')
nanoarrow_dep = nanoarrow_proj.get_variable('nanoarrow_dep')
nanoarrow_ipc_dep = nanoarrow_proj.get_variable('nanoarrow_ipc_dep')
nanoarrow_device_dep = nanoarrow_proj.get_variable('nanoarrow_device_dep')

py = import('python').find_installation(pure: false)

vendored_files = custom_target(
    'generate-pyx',
    output: [
        'nanoarrow.c',
        'nanoarrow_c.pxd',
        'nanoarrow_device.c',
        'nanoarrow_device_c.pxd',
        'nanoarrow_device.h',
        'nanoarrow_device.hpp',
        'nanoarrow.h',
        'nanoarrow.hpp',
        'nanoarrow_ipc.c',
        'nanoarrow_ipc.h',
        'nanoarrow_ipc.hpp',
    ],
    command: [
        py,
        meson.current_source_dir() + '/../../bootstrap.py',
        '--output-dir',
        meson.current_build_dir(),
    ],
)

nanoarrow_pyx_dep = declare_dependency(sources: vendored_files[1])

cyfiles = [
    '_array.pyx',
    '_array_stream.pyx',
    '_buffer.pyx',
    '_device.pyx',
    '_ipc_lib.pyx',
    '_schema.pyx',
    '_types.pyx',
    '_utils.pyx',
]

cython_args = [
    '--include-dir',
    meson.current_source_dir(),
    '--include-dir',
    meson.current_build_dir(),
]
if get_option('debug')
    cython_args += ['--gdb']
endif

fs = import('fs')
foreach cyf : cyfiles
    cyfile_deps = [nanoarrow_pyx_dep, nanoarrow_dep]

    stem = fs.stem(cyf)
    if stem in ['_array', '_device']
        cyfile_deps += [nanoarrow_device_dep]
    elif stem == '_ipc_lib'
        cyfile_deps += [nanoarrow_ipc_dep, flatcc_dep, zstd_dep]
    endif

    py.extension_module(
        stem,
        sources: [cyf],
        cython_args: cython_args,
        dependencies: cyfile_deps,
        subdir: 'nanoarrow',
        install: true,
    )
endforeach

py_sources = [
    '__init__.py',
    '_static_version.py',
    '_version.py',
    'array.py',
    'array_stream.py',
    'c_array.py',
    'c_array_stream.py',
    'c_buffer.py',
    'c_schema.py',
    'device.py',
    'extension.py',
    'extension_canonical.py',
    'ipc.py',
    'iterator.py',
    '_repr_utils.py',
    'schema.py',
    'visitor.py',
]

foreach source : py_sources
    py.install_sources(source, subdir: 'nanoarrow')
endforeach
