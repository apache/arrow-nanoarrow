# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

if [ -z "$NANOARROW_ARROW_TESTING_DIR" ]; then
  echo "NANOARROW_ARROW_TESTING_DIR env is not set"
  exit 1
fi

INTEGRATION_1_0_0="${NANOARROW_ARROW_TESTING_DIR}/data/arrow-ipc-stream/integration/1.0.0-littleendian"
JSON_GZ_FILES=$(find "${INTEGRATION_1_0_0}" -name "*.json.gz")
N_FAIL=0

for json_gz_file in ${JSON_GZ_FILES} ; do
  json_file=$(echo "${json_gz_file}" | sed -e s/.json.gz/.json/)
  ipc_file=$(echo "${json_gz_file}" | sed -e s/.json.gz/.stream/)
  json_gz_label=$(basename ${json_gz_file})
  ipc_label=$(basename ${ipc_file})

  # Unzip the .json.gz file
  gzip --decompress -c "${json_gz_file}" > "${json_file}"

  # Skip dictionary test files for now to keep the noise down
  if echo "${json_gz_file}" | grep -e "dictionary" >/dev/null; then
    echo "[SKIP] ${json_gz_label}"
    continue
  fi

  # Read IPC, check against IPC
  ./integration_test_util \
      --from ipc "${ipc_file}" \
      --check ipc "${ipc_file}"

  if [ $? -eq 0 ]; then
    echo "[PASS] ${ipc_label} --check ${ipc_label}"
  else
    echo "[FAIL] ${ipc_label} --check ${ipc_label}"
    N_FAIL=$((N_FAIL+1))
  fi

  # Read JSON, check against JSON
  ./integration_test_util \
      --from json "${json_file}" \
      --check json "${json_file}"

  if [ $? -eq 0 ]; then
    echo "[PASS] ${json_gz_label} --check ${json_gz_label}"
  else
    echo "[FAIL] ${json_gz_label} --check ${json_gz_label}"
    N_FAIL=$((N_FAIL+1))
  fi

  # Read JSON, check against IPC
  ./integration_test_util \
      --from json "${json_file}" \
      --check ipc "${ipc_file}"

  if [ $? -eq 0 ]; then
    echo "[PASS] ${json_gz_label} --check ${ipc_label}"
  else
    echo "[FAIL] ${json_gz_label} --check ${ipc_label}"
    N_FAIL=$((N_FAIL+1))
  fi

  # Clean up the json file
  rm "${json_file}"
done

exit $N_FAIL
