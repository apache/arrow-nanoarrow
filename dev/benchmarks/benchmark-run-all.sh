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

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

benchmarks_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
build_dir="$1"

if [ -z "${CMAKE_BIN}" ]; then
  CMAKE_BIN=cmake
fi

if [ -z "${build_dir}" ]; then
  build_dir=build
fi

presets=$("${CMAKE_BIN}" --list-presets | grep -e " - " | sed -e "s/^.* //")
pushd "${build_dir}"
for preset in ${presets}; do
    echo "::group::Build ${preset} benchmarks"
    if [ ! -d ${preset} ]; then
        mkdir ${preset}
    fi

    pushd ${preset}
    cmake -S "${benchmarks_source_dir}" --preset ${preset}
    cmake --build .
    popd
done

for preset in ${presets}; do
    echo "::group::Run ${preset} benchmarks"
    pushd ${preset}
    ctest -VV
    popd
done
popd
