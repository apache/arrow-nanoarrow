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

main() {
    pushd "${TARGET_NANOARROW_DIR}"

    # Run doxygen
    pushd src/apidoc
    doxygen
    popd

    pushd docs

    # Use the README as the docs homepage
    pandoc ../README.md --from markdown --to rst -s -o source/README_generated.rst

    # Build sphinx project
    sphinx-build source _build/html

    # Install the R package from source
    R CMD INSTALL ../r --preclean

    # Build R documentation
    Rscript -e 'pkgdown::build_site_github_pages("../r", dest_dir = "../docs/_build/html/r", new_process = FALSE, install = FALSE)'

    popd

    popd
}

# Clean the previous build
rm -rf _build
mkdir -p _build

main >> "${TARGET_NANOARROW_DIR}/docs/_build/build-docs.log" 2>&1
echo "${TARGET_NANOARROW_DIR}/docs/_build/html"
