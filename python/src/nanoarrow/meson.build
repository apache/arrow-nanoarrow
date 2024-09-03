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

# Try to resolve the symlink to a subproject first and fall back
# to the wrap entry if that is unsuccessful. Ideally Meson would
# take care of this for us, but there is a bug upstream
# https://github.com/mesonbuild/meson/issues/13746#issuecomment-2392510954
nanoarrow_proj = subproject('arrow-nanoarrow')
nanoarrow_dep = nanoarrow_proj.get_variable('nanoarrow_dep')
nanoarrow_ipc_dep = nanoarrow_proj.get_variable('nanoarrow_ipc_dep')
nanoarrow_device_dep = nanoarrow_proj.get_variable('nanoarrow_device_dep')

py = import('python').find_installation(pure: false)

generated_pyx = custom_target(
    'generate-pyx',
    output: 'nanoarrow_c.pxd',
    command: [py, meson.current_source_dir() + '/../../bootstrap.py'],
)
nanoarrow_c_dep = declare_dependency(sources: generated_pyx)

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
    meson.project_source_root() / 'vendor',  # for generated nanoarrow_c file
]
if get_option('buildtype') == 'debug'
    cython_args += ['--gdb']
endif

fs = import('fs')
foreach cyf : cyfiles
  cyfile_deps = [nanoarrow_c_dep, nanoarrow_dep]

  stem = fs.stem(cyf)
  if stem in ['_array', '_device']
    cyfile_deps += [nanoarrow_device_dep]
  elif stem == '_ipc_lib'
    cyfile_deps += [nanoarrow_ipc_dep]
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
    'visitor.py',
]

foreach source: py_sources
    py.install_sources(
        source,
        subdir: 'nanoarrow'
    )
endforeach
