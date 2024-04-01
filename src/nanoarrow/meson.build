conf_data = configuration_data()

ns = get_option('NANOARROW_NAMESPACE')
conf_data.set('NANOARROW_NAMESPACE_DEFINE', '#define NANOARROW_NAMESPACE ' + ns)

# TODO: CMake uses a regex to extract major/minor/patch from this string
version = '0.5.0-SNAPSHOT'
ver_major = 0
ver_minor = 5
ver_patch = 0
conf_data.set('NANOARROW_VERSION', version)
conf_data.set('NANOARROW_VERSION_MAJOR', ver_major)
conf_data.set('NANOARROW_VERSION_MINOR', ver_minor)
conf_data.set('NANOARROW_VERSION_PATCH', ver_patch)

configure_file(input: 'nanoarrow_config.h.in',
               output: 'nanoarrow_config.h',
               configuration: conf_data)

nanoarrow = library(
    'nanoarrow',
    'array.c',
    'schema.c',
    'array_stream.c',
    'utils.c',
    install: true)

incdir = include_directories('..')

if get_option('NANOARROW_BUILD_TESTS') or get_option('NANOARROW_BUILD_INTEGRATION_TESTS')
  nlohmann_json_dep = dependency('nlohmann_json')

  c_data_integration_lib = library('nanoarrow_c_data_integration',
                                   'integration/c_data_integration.cc',
                                   link_with: nanoarrow,
                                   dependencies: [nlohmann_json_dep],
                                   include_directories: incdir)

endif

if get_option('NANOARROW_BUILD_TESTS')
  # CMake configuration sets MEMORYCHECK_COMMAND_OPTIONS but with meson you instead
  # wrap the tests with valgrind via `meson test --wrap=valgrind`. See
  # https://mesonbuild.com/Unit-tests.html

  # Similarly code coverage has a built in option users should use instead
  # https://mesonbuild.com/Unit-tests.html#coverage

  if get_option('NANOARROW_ARROW_STATIC')
      arrow_dep = dependency('Arrow', modules: ['Arrow::arrow_static'])
  else
      arrow_dep = dependency('Arrow', modules: ['Arrow::arrow_shared'])
  endif

  gtest_dep = dependency('gtest', fallback: ['gtest', 'gtest_main_dep'])

  # TODO: the CMake configuration sets a different C++ version depending on
  # the version of Arrow used. Meson allows this standard to be set by subproject
  # do we need to do anything here?
  # https://mesonbuild.com/Builtin-options.html#specifying-options-per-subproject

  utils_test = executable('utils_test', 'utils_test.cc',
                          link_with: nanoarrow,
                          dependencies: [arrow_dep, gtest_dep],
                          include_directories: incdir,
                          install: true)
  test('utils test', utils_test)

  buffer_test = executable('buffer_test', 'buffer_test.cc',
                           dependencies: [arrow_dep, gtest_dep],
                          link_with: nanoarrow,
                           include_directories: incdir,
                           install: true)
  test('buffer test', buffer_test)

  array_test = executable('array_test', 'array_test.cc',
                          dependencies: [arrow_dep, gtest_dep],
                          link_with: nanoarrow,
                          include_directories: incdir,
                          install: true)
  test('array test', array_test)

  schema_test = executable('schema_test', 'schema_test.cc',
                           dependencies: [arrow_dep, gtest_dep],
                          link_with: nanoarrow,
                           include_directories: incdir,
                           install: true)
  test('schema test', schema_test)

  array_stream_test = executable('array_stream_test', 'array_stream_test.cc',
                                 dependencies: [arrow_dep, gtest_dep],
                                 link_with: nanoarrow,
                                 include_directories: incdir,
                                 install: true)
  test('array_stream test', array_stream_test)

  nanoarrow_hpp_test = executable('nanoarrow_hpp_test', 'nanoarrow_hpp_test.cc',
                                  dependencies: [arrow_dep, gtest_dep],
                                  link_with: nanoarrow,
                                  include_directories: incdir,
                                  install: true)
  test('nanoarrow_hpp test', nanoarrow_hpp_test)

  nlohmann_json_dep = dependency('nlohmann_json')
  nanoarrow_testing_test = executable('nanoarrow_testing_test', 'nanoarrow_testing_test.cc',
                                      dependencies: [arrow_dep, gtest_dep, nlohmann_json_dep],
                                      link_with: nanoarrow,
                                      include_directories: incdir,
                                      install: true)
  test('nanoarrow_testing test', nanoarrow_testing_test)


  c_data_integration_test = executable('c_data_integration_test', 'integration/c_data_integration_test.cc',
                                       link_with: c_data_integration_lib,
                                       dependencies: [arrow_dep, gtest_dep],
                                       include_directories: incdir,
                                       install: true)
  test('c_data_integration test', c_data_integration_test)


endif
