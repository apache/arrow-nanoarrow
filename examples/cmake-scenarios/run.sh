#!/usr/bin/env bash
#
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

set -exuo pipefail

# For Windows, the PATH needs to be set for it to be able to find the right DLLs
WIN_DLL_NANOARROW_FETCHED="$(pwd)/scratch/build_against_fetched_shared/_deps/nanoarrow-build/Debug"
WIN_DLL_NANOARROW_BUILT="$(pwd)/scratch/nanoarrow_build/Debug"
WIN_DLL_NANOARROW_INSTALLED="$(pwd)/scratch/nanoarrow_install/bin"

for dir in scratch/build*; do
    # Special cases where we have to set PATH on Windows
    if [ "${dir}" = "scratch/build_against_fetched_shared" ] && [ "${OSTYPE}" = "msys" ]; then
        PATH="${PATH}:${WIN_DLL_NANOARROW_FETCHED}"  ./${dir}/Debug/minimal_cpp_app
    elif [ "${dir}" = "scratch/build_shared" ] && [ "${OSTYPE}" = "msys" ]; then
        PATH="${PATH}:${WIN_DLL_NANOARROW_BUILT}" ./${dir}/Debug/minimal_cpp_app
    elif [ "${dir}" = "scratch/build_against_install_shared" ] && [ "${OSTYPE}" = "msys" ]; then
        PATH="${PATH}:${WIN_DLL_NANOARROW_INSTALLED}" ./${dir}/Debug/minimal_cpp_app
    elif [ "${OSTYPE}" = "msys" ]; then
        ./${dir}/Debug/minimal_cpp_app
    else
        ./${dir}/minimal_cpp_app
    fi
done

echo "Success!"
