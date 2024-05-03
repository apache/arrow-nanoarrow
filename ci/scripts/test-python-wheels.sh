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
  0) export CIBW_BUILD=""
     ;;
  1) export CIBW_BUILD="$1"
     ;;
  *) echo "Usage:"
     echo "  Build and test wheels locally using cibuildwheel"
     echo "    (requires pip install cibuildwheel)"
     echo "    $0  ...builds all wheels"
     echo "    $0 'pp*' ...builds and tests just pypy wheels"
     exit 1
     ;;
esac

# Respect existing CIBW_TEST_REQUIRES (could be used to test all wheels
# with numpy and pyarrow installed, for example)
export CIBW_TEST_REQUIRES="pytest $CIBW_TEST_REQUIRES"
export CIBW_TEST_COMMAND="pytest {package}/tests -vv"

pushd "${NANOARROW_DIR}"
python -m cibuildwheel --output-dir python/dist python
popd
