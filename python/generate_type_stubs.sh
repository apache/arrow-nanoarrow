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

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Generates src/nanoarrow/*.pxi given the currently installed copy of
# nanoarrow. Requires mypy and black (where black is pinned to the
# same version as in pre-commit)

pushd "${SOURCE_DIR}"

# We'll add the license to the start of the files
LICENSE='
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
'

# Remove old stubs
find src/nanoarrow -name "*.pyi" -delete

# Generate new ones
pushd src/nanoarrow
for mod in $(find . -name "*.pyx" | sed -e "s|./||" -e "s|.pyx||"); do
  stubgen --module "nanoarrow.${mod}" --include-docstrings -o ../../build/tmp
  echo "$LICENSE" > "${mod}.pyi"
  cat "../../build/tmp/nanoarrow/${mod}.pyi" >> "${mod}.pyi"
  black "${mod}.pyi"
done
popd

popd
