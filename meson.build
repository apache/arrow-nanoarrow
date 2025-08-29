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
    'c',
    'cpp',
    version: '0.8.0-SNAPSHOT',
    license: 'Apache-2.0',
    meson_version: '>=0.58.0',
    default_options: ['c_std=c99', 'warning_level=2', 'cpp_std=c++17'],
)

cc = meson.get_compiler('c')
add_project_arguments(
    cc.get_supported_arguments(['-Wno-misleading-indentation']),
    language: 'c',
)
cpp = meson.get_compiler('cpp')
add_project_arguments(
    cpp.get_supported_arguments(['-Wno-misleading-indentation']),
    language: 'cpp',
)

if get_option('debug')
    add_project_arguments('-DNANOARROW_DEBUG', language: 'c')
    add_project_arguments('-DNANOARROW_DEBUG', language: 'cpp')
endif

nanoarrow_dep_args = []
if host_machine.system() == 'windows' and get_option('default_library') == 'shared'
    add_project_arguments(
        ['-DNANOARROW_BUILD_DLL', '-DNANOARROW_EXPORT_DLL'],
        language: 'c',
    )
    add_project_arguments(
        ['-DNANOARROW_BUILD_DLL', '-DNANOARROW_EXPORT_DLL'],
        language: 'cpp',
    )
    nanoarrow_dep_args += ['-DNANOARROW_BUILD_DLL']
endif

subdir('src/nanoarrow')
incdir = include_directories('src/')

install_headers(
    'src/nanoarrow/nanoarrow.h',
    'src/nanoarrow/nanoarrow.hpp',
    subdir: 'nanoarrow',
)

install_headers(
    'src/nanoarrow/hpp/array_stream.hpp',
    'src/nanoarrow/hpp/buffer.hpp',
    'src/nanoarrow/hpp/exception.hpp',
    'src/nanoarrow/hpp/operators.hpp',
    'src/nanoarrow/hpp/unique.hpp',
    'src/nanoarrow/hpp/view.hpp',
    subdir: 'nanoarrow/hpp',
)

install_headers(
    'src/nanoarrow/common/inline_array.h',
    'src/nanoarrow/common/inline_buffer.h',
    'src/nanoarrow/common/inline_types.h',
    subdir: 'nanoarrow/common',
)

nanoarrow_lib = library(
    'nanoarrow',
    'src/nanoarrow/common/array.c',
    'src/nanoarrow/common/schema.c',
    'src/nanoarrow/common/array_stream.c',
    'src/nanoarrow/common/utils.c',
    include_directories: [incdir],
    install: true,
    gnu_symbol_visibility: 'hidden',
)

pkg = import('pkgconfig')
pkg.generate(
    nanoarrow_lib,
    description: 'Helpers for Arrow C Data & Arrow C Stream interfaces',
)

nanoarrow_dep = declare_dependency(
    include_directories: [incdir],
    link_with: nanoarrow_lib,
    compile_args: nanoarrow_dep_args,
)

meson.override_dependency('nanoarrow', nanoarrow_dep)

if get_option('ipc').enabled()
    # flatcc does not export symbols in a way that works with
    # MSVC compilers, so we force static linkage
    if host_machine.system() == 'windows'
        flatcc_dep = dependency(
            'flatcc',
            default_options: ['default_library=static'],
        )
    else
        flatcc_dep = dependency('flatcc')
    endif
    ipc_lib_deps = [nanoarrow_dep, flatcc_dep]
    ipc_lib_c_args = []

    if get_option('ipc_with_zstd').enabled()
        zstd_dep = dependency('libzstd')
        ipc_lib_deps += zstd_dep
        ipc_lib_c_args += '-DNANOARROW_IPC_WITH_ZSTD'
    endif

    install_headers(
        'src/nanoarrow/nanoarrow_ipc.h',
        'src/nanoarrow/ipc/flatcc_generated.h',
        subdir: 'nanoarrow',
    )

    nanoarrow_ipc_lib = library(
        'nanoarrow_ipc',
        'src/nanoarrow/ipc/codecs.c',
        'src/nanoarrow/ipc/decoder.c',
        'src/nanoarrow/ipc/encoder.c',
        'src/nanoarrow/ipc/reader.c',
        'src/nanoarrow/ipc/writer.c',
        dependencies: ipc_lib_deps,
        install: true,
        c_args: ipc_lib_c_args,
        gnu_symbol_visibility: 'hidden',
    )
    pkg.generate(nanoarrow_ipc_lib, filebase: 'nanoarrow-ipc')

    nanoarrow_ipc_dep = declare_dependency(
        include_directories: [incdir],
        link_with: nanoarrow_ipc_lib,
        dependencies: [nanoarrow_dep],
    )
    meson.override_dependency('nanoarrow-ipc', nanoarrow_ipc_dep)
    zlib_dep = dependency('zlib', default_options: ['tests=disabled'])
else
    flatcc_dep = disabler()
    nanoarrow_ipc_dep = disabler()
    zlib_dep = disabler()
endif

needs_metal = false
metal_dep = disabler()
metal_cpp_dep = disabler()
if get_option('metal').enabled()
    if host_machine.system() == 'darwin'
        needs_metal = true
        metal_dep = dependency(
            'appleframeworks',
            modules: ['Foundation', 'Metal'],
        )
        metal_cpp_dep = dependency('metal-cpp')
    else
        warning('metal option enabled on a non-darwin platform - ignored!')
    endif
endif

needs_device = get_option('device').enabled() or get_option('cuda').enabled() or needs_metal

if needs_device
    device_deps = [nanoarrow_dep]
    device_srcs = ['src/nanoarrow/device/device.c']
    device_defines = []

    if needs_metal
        device_deps += metal_dep
        device_deps += metal_cpp_dep
        device_srcs += 'src/nanoarrow/device/metal.cc'
        device_defines += '-DNANOARROW_DEVICE_WITH_METAL'
    endif

    if get_option('cuda').enabled()
        warning('cuda option enabled but not implemented in Meson config!')
    endif

    install_headers('src/nanoarrow/nanoarrow_device.h', subdir: 'nanoarrow')

    nanoarrow_device_lib = library(
        'nanoarrow_device',
        sources: device_srcs,
        dependencies: device_deps,
        install: true,
        cpp_args: device_defines,
        gnu_symbol_visibility: 'hidden',
    )
    pkg.generate(nanoarrow_device_lib, filebase: 'nanoarrow-device')

    nanoarrow_device_dep = declare_dependency(
        include_directories: [incdir],
        link_with: nanoarrow_device_lib,
        dependencies: device_deps,
    )
    meson.override_dependency('nanoarrow-device', nanoarrow_device_dep)
else
    nanoarrow_device_dep = disabler()
endif

needs_testing = get_option('testing').enabled() or get_option('tests').enabled()
if needs_testing
    nlohmann_json_dep = dependency('nlohmann_json')

    nanoarrow_testing_lib = library(
        'nanoarrow_testing',
        sources: ['src/nanoarrow/testing/testing.cc'],
        dependencies: [nanoarrow_dep, nlohmann_json_dep],
        include_directories: incdir,
        install: true,
        gnu_symbol_visibility: 'hidden',
    )
    pkg.generate(nanoarrow_testing_lib, filebase: 'nanoarrow-testing')

    nanoarrow_testing_dep = declare_dependency(
        include_directories: [incdir],
        link_with: nanoarrow_testing_lib,
        dependencies: [nanoarrow_dep, nlohmann_json_dep],
    )
    meson.override_dependency('nanoarrow-testing', nanoarrow_testing_dep)
else
    nanoarrow_testing_dep = disabler()
endif

if get_option('tests').enabled() or get_option('integration_tests').enabled()
    c_data_integration_lib = library(
        'nanoarrow_c_data_integration',
        'src/nanoarrow/integration/c_data_integration.cc',
        dependencies: [nanoarrow_testing_dep, nanoarrow_dep],
        include_directories: incdir,
        gnu_symbol_visibility: 'hidden',
    )

    c_data_integration_dep = declare_dependency(
        link_with: [c_data_integration_lib],
    )
else
    c_data_integration_dep = disabler()
endif

test_cpp_args = []
if get_option('tests').enabled()
    # CMake configuration sets MEMORYCHECK_COMMAND_OPTIONS but with meson you instead
    # wrap the tests with valgrind via `meson test --wrap=valgrind`. See
    # https://mesonbuild.com/Unit-tests.html

    # Similarly code coverage has a built in option users should use instead
    # https://mesonbuild.com/Unit-tests.html#coverage

    arrow_dep = dependency('arrow', include_type: 'system')
    test_cpp_args += ['-DNANOARROW_BUILD_TESTS_WITH_ARROW']
    gtest_dep = dependency('gtest_main')
    gmock_dep = dependency('gmock')
else
    arrow_dep = disabler()
    gtest_dep = disabler()
    gmock_dep = disabler()
endif

nanoarrow_tests = ['utils', 'buffer', 'array', 'schema', 'array-stream']

foreach name : nanoarrow_tests
    exc = executable(
        name + '-test',
        sources: 'src/nanoarrow/common/' + name.replace('-', '_') + '_test.cc',
        include_directories: incdir,
        dependencies: [nanoarrow_testing_dep, arrow_dep, gtest_dep, gmock_dep],
        cpp_args: test_cpp_args,
    )
    test(name, exc)
endforeach

nanoarrow_hpp_tests = ['array_stream', 'buffer', 'exception', 'unique', 'view']

foreach name : nanoarrow_hpp_tests
    exc = executable(
        'hpp-' + name + '-test',
        sources: 'src/nanoarrow/hpp/' + name.replace('-', '_') + '_test.cc',
        include_directories: incdir,
        dependencies: [nanoarrow_testing_dep, gtest_dep, gmock_dep],
    )
    test(name, exc)
endforeach

testing_test = executable(
    'nanoarrow-testing-test',
    'src/nanoarrow/testing/testing_test.cc',
    include_directories: incdir,
    dependencies: [nanoarrow_testing_dep, arrow_dep, gtest_dep],
)

c_data_integration_test = executable(
    'c-data-integration-test',
    'src/nanoarrow/integration/c_data_integration_test.cc',
    dependencies: [
        c_data_integration_dep,
        nanoarrow_testing_dep,
        arrow_dep,
        gtest_dep,
    ],
    include_directories: incdir,
)
test('c-data-integration', c_data_integration_test)

ipc_test_files = {
    'ipc-decoder': {
        'src': 'decoder',
        'deps': [nanoarrow_ipc_dep, flatcc_dep, arrow_dep, gtest_dep, gmock_dep],
        'timeout': 30,
    },
    'ipc-reader': {
        'src': 'reader',
        'deps': [nanoarrow_ipc_dep, arrow_dep, gtest_dep],
        # the ipc reader test can take longer when executed
        # under valgrind, hence the increased timeout
        'timeout': 90,
    },
    'ipc-files': {
        'src': 'files',
        'deps': [
            nanoarrow_testing_dep,
            nanoarrow_ipc_dep,
            flatcc_dep,
            zlib_dep,
            arrow_dep,
            gtest_dep,
        ],
        'timeout': 30,
    },
    'ipc-hpp': {
        'src': 'ipc_hpp',
        'deps': [nanoarrow_testing_dep, nanoarrow_ipc_dep, gtest_dep],
        'timeout': 30,
    },
}

foreach name, config : ipc_test_files
    exc = executable(
        'nanoarrow-' + name + '-test',
        'src/nanoarrow/ipc/' + config['src'] + '_test.cc',
        dependencies: config['deps'],
    )
    test(name, exc, timeout: config['timeout'])
endforeach

device_tests = ['device', 'device_hpp']
foreach device_test : device_tests
    exc = executable(
        'nanoarrow-' + device_test.replace('_', '-') + '-test',
        'src/nanoarrow/device/' + device_test + '_test.cc',
        dependencies: [nanoarrow_device_dep, nanoarrow_testing_dep, gtest_dep],
    )
    test(device_test.replace('_', '-'), exc)
endforeach

exc = executable(
    'nanoarrow-device-metal-test',
    'src/nanoarrow/device/metal_test.cc',
    dependencies: [
        nanoarrow_device_dep,
        nanoarrow_testing_dep,
        gtest_dep,
        metal_cpp_dep,
    ],
)
test('nanoarrow-device-metal', exc)

if get_option('benchmarks').enabled()
    subdir('dev/benchmarks')
endif


if get_option('apps').enabled()
    executable(
        'dump_stream',
        'src/apps/dump_stream.c',
        dependencies: [nanoarrow_ipc_dep],
    )
endif
