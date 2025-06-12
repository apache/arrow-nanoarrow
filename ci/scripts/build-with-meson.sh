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

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
NANOARROW_DIR="$(cd "${SOURCE_DIR}/../.." && pwd)"

show_header() {
  if [ -z "$GITHUB_ACTIONS" ]; then
    echo ""
    printf '=%.0s' $(seq ${#1}); printf '\n'
    echo "${1}"
    printf '=%.0s' $(seq ${#1}); printf '\n'
  else
    echo "::group::${1}"; printf '\n'
  fi
}


case $# in
  0) TARGET_NANOARROW_DIR="${NANOARROW_DIR}"
     ;;
  1) TARGET_NANOARROW_DIR="$1"
     ;;
  *) echo "Usage:"
     echo "  Build nanoarrow based on a source checkout elsewhere:"
     echo "    $0 path/to/arrow-nanoarrow"
     exit 1
     ;;
esac

function main() {
    pushd ${TARGET_NANOARROW_DIR}

    SANDBOX_DIR="_meson_builddir"
    if [ -d "${SANDBOX_DIR}" ]; then
        rm -rf "${SANDBOX_DIR}"
    fi
    mkdir "${SANDBOX_DIR}"

    show_header "Compile project with meson"
    meson setup "${SANDBOX_DIR}" --pkg-config-path $PKG_CONFIG_PATH -Dwerror=true

    pushd "${SANDBOX_DIR}"

    show_header "Run ASAN/UBSAN test suite"
    meson configure \
          -Dbuildtype=debugoptimized \
          -Db_sanitize="address,undefined" \
          -Dtests=enabled \
          -Dtests_with_arrow=enabled \
          -Dipc=enabled \
          -Ddevice=enabled \
          -Dbenchmarks=disabled \
          -Db_coverage=false

    meson compile
    meson test --print-errorlogs

    show_header "Run valgrind test suite"
    meson configure \
          -Dbuildtype=debugoptimized \
          -Db_sanitize=none \
          -Dtests=enabled \
          -Dtests_with_arrow=enabled \
          -Dipc=enabled \
          -Ddevice=enabled \
          -Dbenchmarks=disabled \
          -Db_coverage=false
    meson compile
    meson test --wrap='valgrind --track-origins=yes --leak-check=full' --print-errorlog

    show_header "Run benchmarks"
    meson configure \
          -Dbuildtype=release \
          -Db_sanitize=none \
          -Dtests=disabled \
          -Dtests_with_arrow=enabled \
          -Dipc=enabled \
          -Ddevice=enabled \
          -Dbenchmarks=enabled \
          -Db_coverage=false
    meson compile
    meson test --benchmark --print-errorlogs

    show_header "Run coverage test suite"
    meson configure \
          -Dbuildtype=release \
          -Db_sanitize=none \
          -Dtests=enabled \
          -Dtests_with_arrow=enabled \
          -Dipc=enabled \
          -Ddevice=enabled \
          -Dbenchmarks=disabled \
          -Db_coverage=true

    meson compile
    meson test --print-errorlogs

    show_header "Generate coverage reports"
    ninja coverage
    cat meson-logs/coverage.txt
    popd

    # Clean up subprojects and build folder
    rm -rf "${SANDBOX_DIR}"
    rm -rf "${SUBPROJ_DIR}"

    popd
}

main
