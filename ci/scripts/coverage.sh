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

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
NANOARROW_SOURCE_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

# Use a sandbox dir for intermediate results. This also makes it easier
# to debug the process.
SANDBOX_DIR="${NANOARROW_SOURCE_DIR}/coverage"
if [ -d "${SANDBOX_DIR}" ]; then
    rm -rf "${SANDBOX_DIR}"
fi
mkdir "${SANDBOX_DIR}"

# Run tests with gcov for core library
mkdir "${SANDBOX_DIR}/build"
pushd "${SANDBOX_DIR}/build"

popd

# Run tests with gcov for IPC extension
mkdir "${SANDBOX_DIR}/build-ipc"
pushd "${SANDBOX_DIR}/build-ipc"

popd

# Run covr::package_coverage() on the R package



mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DNANOARROW_BUILD_TESTS=ON
cmake --build .
