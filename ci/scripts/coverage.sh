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
     echo "  Build documentation based on a source checkout elsewhere:"
     echo "    $0 path/to/arrow-nanoarrow"
     echo "  Build documentation for this nanoarrow checkout:"
     echo "    $0"
     exit 1
     ;;
esac

maybe_activate_venv() {
  if [ ! -z "${NANOARROW_PYTHON_VENV}" ]; then
    source "${NANOARROW_PYTHON_VENV}/bin/activate"
  fi
}

function main() {
    maybe_activate_venv

    SANDBOX_DIR="${TARGET_NANOARROW_DIR}/_coverage"
    if [ -d "${SANDBOX_DIR}" ]; then
        rm -rf "${SANDBOX_DIR}"
    fi
    mkdir "${SANDBOX_DIR}"

    # Bulid + run tests with gcov
    show_header "Build + test nanoarrow"
    mkdir "${SANDBOX_DIR}/nanoarrow"
    pushd "${SANDBOX_DIR}/nanoarrow"

    cmake "${TARGET_NANOARROW_DIR}" \
          -DNANOARROW_DEVICE=ON -DNANOARROW_IPC=ON -DNANOARROW_IPC_WITH_ZSTD=ON \
          -DNANOARROW_IPC_WITH_LZ4=ON \
          -DNANOARROW_BUILD_TESTS=ON -DNANOARROW_BUILD_TESTS_WITH_ARROW=ON \
          -DNANOARROW_CODE_COVERAGE=ON
    cmake --build .
    CTEST_OUTPUT_ON_FAILURE=1 ctest .

    popd

    pushd "${SANDBOX_DIR}"

    # Generate coverage.info file for both cmake projects using gcovr
    show_header "Calculate CMake project coverage"
    gcovr -r . -f "${TARGET_NANOARROW_DIR}/src" \
          -e ".*generated\.h" \
          -e ".*_test.cc" \
          --lcov coverage.info

    # Generate the html coverage while we're here
    mkdir html
    gcovr -r . -f "${TARGET_NANOARROW_DIR}/src" \
          -e ".*generated\.h" \
          -e ".*_test.cc" \
       --html html/coverage.html

    # Stripping the leading /nanoarrow/ out of the path is probably possible with
    # an argument of gcovr but none of the obvious ones seem to work so...
    sed -i.bak coverage.info -e 's|SF:/nanoarrow/|SF:|'
    rm coverage.info.bak

    # Print a summary
    show_header "CMake project coverage summary"
    gcovr -s -r . -f "${TARGET_NANOARROW_DIR}/src" \
          -e ".*generated\.h" \
          -e ".*_test.cc"

    # Clean up the build directories
    rm -rf nanoarrow
    rm -rf nanoarrow_ipc

    popd

    # Build + test R package
    show_header "Build + test R package"
    pushd "${SANDBOX_DIR}"
    TARGET_NANOARROW_R_DIR="${TARGET_NANOARROW_DIR}/r" \
        Rscript -e 'saveRDS(covr::package_coverage(Sys.getenv("TARGET_NANOARROW_R_DIR"), relative_path = "/nanoarrow/"), "r_coverage.rds")'
    Rscript -e 'covr:::to_codecov(readRDS("r_coverage.rds")) |> brio::write_file("r_coverage.json")'

    show_header "R package coverage summary"
    Rscript -e 'library(covr); print(readRDS("r_coverage.rds"))'
    popd

    # Build + test Python package with cython/gcc coverage options
    show_header "Build + test Python package"
    pushd "${SANDBOX_DIR}"
    TARGET_NANOARROW_PYTHON_DIR="${TARGET_NANOARROW_DIR}/python"

    pushd "${TARGET_NANOARROW_PYTHON_DIR}"
    python -m pip install -Csetup-args="-Db_coverage=true" .

    # Run tests + coverage.py (generates .coverage with absolute file paths)
    python -m pytest --cov tests/

    # Generate HTML report (file paths not important since it's just for viewing)
    python -m coverage html
    mv htmlcov "${SANDBOX_DIR}/python_htmlcov"

    # Move .coverage to the root directory and generate coverage.xml
    # (generates relative file paths from the root of the repo)
    mv .coverage ..
    cp .coveragerc ..
    pushd ..
    python -m coverage xml
    mv coverage.xml "${SANDBOX_DIR}/python_coverage.xml"
    mv .coverage "${SANDBOX_DIR}/python_coverage.db"
    rm .coveragerc
    popd

    popd
    popd
}

main
