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

export CMAKE_CONFIGURATION_TYPES="Release;Debug"

# Build nanoarrow statically.
cmake -S ../.. -B scratch/nanoarrow_build/ \
    -DCMAKE_INSTALL_PREFIX=scratch/nanoarrow_install/ \
    -DNANOARROW_IPC=ON -DNANOARROW_DEVICE=ON -DNANOARROW_TESTING=ON
cmake --build scratch/nanoarrow_build/
cmake --install scratch/nanoarrow_build/

for nanoarrow_build_type in static shared; do
    # Build the project against the built nanoarrow.
    cmake -S . -B scratch/build_${nanoarrow_build_type}/ \
        -Dnanoarrow_ROOT=scratch/nanoarrow_build \
        -DTEST_BUILD_TYPE=${nanoarrow_build_type}
    cmake --build scratch/build_${nanoarrow_build_type}/

    # Build the project against the installed nanoarrow.
    cmake -S . -B scratch/build_against_install_${nanoarrow_build_type}/ \
        -Dnanoarrow_ROOT=scratch/nanoarrow_install \
        -DTEST_BUILD_TYPE=${nanoarrow_build_type}
    cmake --build scratch/build_against_install_${nanoarrow_build_type}/

    # Now try using FetchContent to get nanoarrow from remote.
    cmake -S . -B scratch/build_against_fetched_${nanoarrow_build_type}/ \
        -DFIND_NANOARROW=OFF \
        -DTEST_BUILD_TYPE=${nanoarrow_build_type}
    cmake --build scratch/build_against_fetched_${nanoarrow_build_type}/
done
