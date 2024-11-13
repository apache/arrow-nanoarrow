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

conf_data = configuration_data()

ns = get_option('namespace')
conf_data.set('NANOARROW_NAMESPACE_DEFINE', '#define NANOARROW_NAMESPACE ' + ns)

version = meson.project_version()
# Remove any pre-release / build identifiers
version_no_pre_release = version.split('-')[0]
version_no_build = version_no_pre_release.split('+')[0]
components = version_no_build.split('.')
assert(
    components.length() >= 3,
    'The version does not contain major, minor and patch',
)
ver_major = components[0]
ver_minor = components[1]
ver_patch = components[2]
conf_data.set('NANOARROW_VERSION', version)
conf_data.set('NANOARROW_VERSION_MAJOR', ver_major)
conf_data.set('NANOARROW_VERSION_MINOR', ver_minor)
conf_data.set('NANOARROW_VERSION_PATCH', ver_patch)

configure_file(
    input: 'nanoarrow_config.h.in',
    output: 'nanoarrow_config.h',
    configuration: conf_data,
    install: true,
    install_dir: get_option('includedir') / 'nanoarrow',
)
