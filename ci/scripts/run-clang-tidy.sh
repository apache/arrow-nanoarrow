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

main() {
    local -r source_dir="${1}"
    local -r build_dir="${2}"

    if [ $(uname) = "Darwin" ]; then
      local -r jobs=$(sysctl -n hw.ncpu)
    else
      local -r jobs=$(nproc)
    fi

    set -x

    run-clang-tidy -p "${build_dir}" -j$jobs \
        -extra-arg=-Wno-unknown-warning-option | \
        tee "${build_dir}/clang-tidy-output.txt"

    if grep -e "warning:" -e "error:" "${build_dir}/clang-tidy-output.txt"; then
      echo "Warnings or errors found!"
      exit 1
    else
      echo "No warnings or errors found!"
    fi

    set +x
}

main "$@"
