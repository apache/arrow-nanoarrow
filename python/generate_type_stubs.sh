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

set -euo pipefail

# Generates src/nanoarrow/*.pyi given the currently installed copy of
# nanoarrow. See requirements-stubs.txt for the regeneration environment.

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GENERATED_DIR="$(mktemp -d)"
trap 'rm -rf "$GENERATED_DIR"' EXIT

cd "${SOURCE_DIR}/src/nanoarrow"
shopt -s nullglob

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

modules=()
sources=(*.pyx)
if (( ${#sources[@]} == 0 )); then
  echo "No Cython sources found" >&2
  exit 1
fi
for source in "${sources[@]}"; do
  modules+=("nanoarrow.${source%.pyx}")
done
mkdir "$GENERATED_DIR/raw"
python "$SOURCE_DIR/generate_type_stubs.py" "$GENERATED_DIR/raw" "${modules[@]}"

# Only replace the source stubs after generation and formatting succeed.
for source in "${sources[@]}"; do
  stub="${source%.pyx}.pyi"
  {
    printf '%s\n' "$LICENSE"
    cat "$GENERATED_DIR/raw/$stub"
  } > "$GENERATED_DIR/$stub"
done
black "$GENERATED_DIR"/*.pyi

# All .pyi files in this directory are generated. Remove stubs for deleted modules.
rm -f -- ./*.pyi
cp "$GENERATED_DIR"/*.pyi .
