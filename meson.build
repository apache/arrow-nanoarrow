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
    version: '0.6.0-SNAPSHOT',
    license: 'Apache 2.0',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c99',
        'warning_level=2',
        'cpp_std=c++17',
    ]
)

# Windows shared libraries are not exporting the right symbols
# See https://github.com/mesonbuild/wrapdb/pull/1536#issuecomment-2136011408
# and https://github.com/apache/arrow-nanoarrow/issues/495
if host_machine.system() == 'windows'
    libtype = 'static_library'
else
    libtype = 'library'
endif

subdir('src/nanoarrow')
incdir = include_directories('src/')

nanoarrow_lib = build_target(
    'nanoarrow',
    'src/nanoarrow/array.c',
    'src/nanoarrow/schema.c',
    'src/nanoarrow/array_stream.c',
    'src/nanoarrow/utils.c',
    include_directories: [incdir],
    install: true,
    target_type: libtype,
)

nanoarrow_dep = declare_dependency(include_directories: [incdir],
                                   link_with: nanoarrow_lib)

if get_option('ipc')
    flatcc_dep = dependency('flatcc')

    nanoarrow_ipc_lib = build_target(
        'nanoarrow_ipc',
        'src/nanoarrow/nanoarrow_ipc_decoder.c',
        'src/nanoarrow/nanoarrow_ipc_reader.c',
        dependencies: [nanoarrow_dep, flatcc_dep],
        install: true,
        target_type: libtype,
    )
    nanoarrow_ipc_dep = declare_dependency(include_directories: [incdir],
                                           link_with: nanoarrow_ipc_lib,
                                           dependencies: [nanoarrow_dep, flatcc_dep])

needs_device = get_option('device') or get_option('metal') or get_option('cuda')
if needs_device
  device_deps = [nanoarrow_dep]
  device_srcs = ['src/nanoarrow/nanoarrow_device.c']
  device_defines = []

  if get_option('metal')
    metal_dep = dependency('appleframeworks', modules: ['Foundation', 'Metal'])
    metal_cpp_dep = dependency('metal-cpp')
    device_deps += metal_dep
    device_deps += metal_cpp_dep
    device_srcs += 'src/nanoarrow/nanoarrow_device_metal.cc'
    device_defines += '-DNANOARROW_DEVICE_WITH_METAL'
  endif

  if get_option('cuda')
    error('CUDA support with the Meson build system is not implemented')
  endif

  nanoarrow_device_lib = build_target(
      'nanoarrow_device',
      sources: device_srcs,
      dependencies: device_deps,
      install: true,
      target_type: libtype,
      cpp_args: device_defines,
  )
endif

if get_option('tests') or get_option('integration_tests')
  nlohmann_json_dep = dependency('nlohmann_json')

  c_data_integration_lib = library('nanoarrow_c_data_integration',
                                   'src/nanoarrow/integration/c_data_integration.cc',
                                   link_with: nanoarrow_lib,
                                   dependencies: [nlohmann_json_dep],
                                   include_directories: incdir)

endif

if get_option('tests')
  # CMake configuration sets MEMORYCHECK_COMMAND_OPTIONS but with meson you instead
  # wrap the tests with valgrind via `meson test --wrap=valgrind`. See
  # https://mesonbuild.com/Unit-tests.html

  # Similarly code coverage has a built in option users should use instead
  # https://mesonbuild.com/Unit-tests.html#coverage

  arrow_dep = dependency('arrow')
  gtest_dep = dependency('gtest_main')
  gmock_dep = dependency('gmock')
  nlohmann_json_dep = dependency('nlohmann_json')

  nanoarrow_tests = {
      'utils': {
          'deps': [arrow_dep, gtest_dep, gmock_dep, nlohmann_json_dep],
      },
      'buffer': {
          'deps': [arrow_dep, gtest_dep],
      },
      'array': {
          'deps': [arrow_dep, gtest_dep, gmock_dep],
      },
      'schema': {
          'deps': [arrow_dep, gtest_dep],
      },
      'array-stream': {
          'deps': [arrow_dep, gtest_dep, gmock_dep],
      },
      'nanoarrow-hpp': {
          'deps': [arrow_dep, gtest_dep, gmock_dep, nlohmann_json_dep],
      },
      'nanoarrow-testing': {
          'deps': [arrow_dep, gtest_dep, nlohmann_json_dep],
      },
  }

  foreach name, config : nanoarrow_tests
    exc = executable(
        name + '-test',
        sources: 'src/nanoarrow/' + name.replace('-', '_') + '_test.cc',
        link_with: nanoarrow_lib,
        include_directories: incdir,
        dependencies: config['deps'],
    )
    test(name, exc)
  endforeach

  c_data_integration_test = executable('c-data-integration-test',
                                       'src/nanoarrow/integration/c_data_integration_test.cc',
                                       link_with: c_data_integration_lib,
                                       dependencies: [arrow_dep, gtest_dep],
                                       include_directories: incdir)
  test('c-data-integration', c_data_integration_test)

  if get_option('ipc')
      zlib_dep = dependency('zlib')
      ipc_test_files = {
          'nanoarrow-ipc-decoder': {
              'deps': [nanoarrow_ipc_dep, arrow_dep, gtest_dep],
              'timeout': 30,
          },
          'nanoarrow-ipc-reader': {
              'deps': [nanoarrow_ipc_dep, arrow_dep, gtest_dep],
              # the ipc reader test can take longer when executed
              # under valgrind, hence the increased timeout
              'timeout': 90,

          },
          'nanoarrow-ipc-files': {
              'deps': [
                  nanoarrow_ipc_dep,
                  zlib_dep,
                  arrow_dep,
                  gtest_dep,
                  nlohmann_json_dep
              ],
              'timeout': 30,
          },
          'nanoarrow-ipc-hpp': {
              'deps': [nanoarrow_ipc_dep, gtest_dep],
              'timeout': 30,
          },
      }

      foreach name, config : ipc_test_files
          exc = executable(
              name + '-test',
              'src/nanoarrow/' + name.replace('-', '_') + '_test.cc',
              dependencies: config['deps']
          )
          test(name, exc, timeout: config['timeout'])
      endforeach
  endif

  if needs_device
    device_tests = ['nanoarrow_device', 'nanoarrow_device_hpp']
    foreach device_test : device_tests
      exc = executable(
          device_test.replace('_', '-') + '-test',
          'src/nanoarrow/' + device_test + '_test.cc',
          link_with: nanoarrow_device_lib,
          dependencies: [nanoarrow_dep, gtest_dep],
      )
      test(device_test.replace('_', '-'), exc)
    endforeach

    if get_option('metal')
      exc = executable(
          'nanoarrow-device-metal-test',
          'src/nanoarrow/nanoarrow_device_metal_test.cc',
          link_with: nanoarrow_device_lib,
          dependencies: [nanoarrow_dep, gtest_dep, metal_cpp_dep],
      )
      test('nanoarrow-device-metal', exc)
    endif
  endif
endif

if get_option('benchmarks')
  subdir('dev/benchmarks')
endif


if get_option('apps')
  if get_option('ipc')
  executable(
      'dump_stream',
      'src/apps/dump_stream.c',
      dependencies: [nanoarrow_dep, nanoarrow_ipc_dep]
  )
  endif
endif
