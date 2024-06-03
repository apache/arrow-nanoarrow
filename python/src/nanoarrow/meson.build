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

nanoarrow_dep = dependency('nanoarrow')

generated_pyx = custom_target(
    'generate-pyx',
    input: meson.current_build_dir(),
    output: 'nanoarrow_c.pxd',
    command: [py, '../../bootstrap.py', '@INPUT@'],
    install: true,
    install_dir: '.',
    #depends: nanoarrow_dep,
    # ERROR: custom_target keyword argument 'depends' was of type
    # array[InternalDependency] but should have been array[BuildTarget |
    # CustomTarget] - seems fragile without this?

)
nanoarrow_c_dep = declare_dependency(sources: generated_pyx)

# TODO: we need dependencies on nanoarrow_ipc and nanoarrow_device,
# which should happen as part of https://github.com/apache/arrow-nanoarrow/pull/483

py.extension_module(
    '_lib',
    sources: ['_lib.pyx'],
    cython_args: cython_args,
    dependencies: [nanoarrow_dep, nanoarrow_c_dep],
    subdir: 'nanoarrow/',
    install: true,
)

py.extension_module(
    '_ipc_lib',
    sources: ['_ipc_lib.pyx'],
    cython_args: cython_args,
    dependencies: [nanoarrow_dep, nanoarrow_c_dep],
    subdir: 'nanoarrow',
    install: true,
)
