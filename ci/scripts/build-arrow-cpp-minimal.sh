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

# Builds a minimal Arrow C++ version required for running nanoarrow tests

set -e
set -o pipefail

case $# in
  2) ARROW_CPP_VERSION="$1"
     ARROW_CPP_INSTALL_DIR="$2"
     ;;
  *) echo "Usage:"
     echo "  Build and install Arrow C++:"
     echo "    $0 <version> <install dir>"
     exit 1
     ;;
esac

# Ensure install directory exists and is absolute
if [ ! -d "${ARROW_CPP_INSTALL_DIR}" ]; then
  mkdir -p "${ARROW_CPP_INSTALL_DIR}"
fi

ARROW_CPP_INSTALL_DIR="$(cd "${ARROW_CPP_INSTALL_DIR}" && pwd)"

ARROW_CPP_SCRATCH_DIR="arrow-cpp-build-${ARROW_CPP_VERSION}"

mkdir "${ARROW_CPP_SCRATCH_DIR}"
pushd "${ARROW_CPP_SCRATCH_DIR}"

curl -L "https://www.apache.org/dyn/closer.lua?action=download&filename=arrow/arrow-${ARROW_CPP_VERSION}/apache-arrow-${ARROW_CPP_VERSION}.tar.gz" | \
  tar -zxf -
mkdir build && cd build
cmake ../apache-arrow-${ARROW_CPP_VERSION}/cpp \
  -DARROW_JEMALLOC=OFF \
  -DARROW_SIMD_LEVEL=NONE \
  -DARROW_FILESYSTEM=OFF \
  -DCMAKE_INSTALL_PREFIX="${ARROW_CPP_INSTALL_DIR}"
cmake --build . --parallel $(nproc)
cmake --install . --prefix="${ARROW_CPP_INSTALL_DIR}"

popd

# On success, we can remove the build directory
rm -rf "${ARROW_CPP_SCRATCH_DIR}"
