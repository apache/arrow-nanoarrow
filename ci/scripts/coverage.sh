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

set -e
set -o pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
NANOARROW_SOURCE_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

# Use a sandbox dir for intermediate results. This also makes it easier
# to debug the process.
SANDBOX_DIR="${NANOARROW_SOURCE_DIR}/_coverage"
if [ -d "${SANDBOX_DIR}" ]; then
    rm -rf "${SANDBOX_DIR}"
fi
mkdir "${SANDBOX_DIR}"

# Bulid + run tests with gcov for core library
mkdir "${SANDBOX_DIR}/nanoarrow"
pushd "${SANDBOX_DIR}/nanoarrow"

cmake "${NANOARROW_SOURCE_DIR}" \
    -DNANOARROW_BUILD_TESTS=ON -DNANOARROW_CODE_COVERAGE=ON
cmake --build .
ctest .

popd

# Build + run tests with gcov for IPC extension
mkdir "${SANDBOX_DIR}/nanoarrow_ipc"
pushd "${SANDBOX_DIR}/nanoarrow_ipc"

cmake "${NANOARROW_SOURCE_DIR}/extensions/nanoarrow_ipc" \
    -DNANOARROW_IPC_BUILD_TESTS=ON -DNANOARROW_IPC_CODE_COVERAGE=ON
cmake --build .
ctest .

popd

pushd "${SANDBOX_DIR}"

# Generate coverage.info file for both cmake projects using lcov
lcov --capture --directory . \
    --exclude "*_test.cc" \
    --exclude "/usr/*" \
    --exclude "*/gtest/*" \
    --exclude "*/flatcc/*" \
    --exclude "*_generated.h" \
    --output-file coverage.info

# Print some debug output
lcov --list coverage.info

# Generate the html coverage while we're here
genhtml coverage.info --output-directory html --prefix "${NANOARROW_SOURCE_DIR}"

popd
