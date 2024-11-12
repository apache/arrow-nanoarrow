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

main() {
   maybe_activate_venv

   pushd "${TARGET_NANOARROW_DIR}"

   # Clean the previous build
   rm -rf docs/_build
   mkdir -p docs/_build

   show_header "Run Doxygen for C library"
   pushd src/apidoc
   doxygen
   popd

   show_header "Build nanoarrow Python"

   pushd python
   pip install .
   popd

   show_header "Build Sphinx project"
   pushd docs


   # Use the README as the docs homepage
   pandoc ../README.md --from markdown --to rst -s -o source/README_generated.rst

   # Use R README as the getting started guide for R
   pandoc ../r/README.md --from markdown --to rst -s -o source/getting-started/r.rst

   # Use Python README as the getting started guide for Python
   pandoc ../python/README.md --from markdown --to rst -s -o source/getting-started/python.rst

   # Do some Markdown -> reST conversion
   for f in $(find source -name "*.md"); do
     fout=$(echo $f | sed -e s/.md/.rst/)
     pandoc $f --from markdown --to rst -s -o $fout
   done

   # Build sphinx project
   sphinx-build -W source _build/html

   show_header "Build R documentation"

   # Install the R package from source
   R CMD INSTALL ../r --preclean

   # Build R documentation
   Rscript -e 'pkgdown::build_site("../r", override = list(destination = "../docs/_build/html/r"), new_process = FALSE, install = FALSE)'

   popd

   popd
}

main
