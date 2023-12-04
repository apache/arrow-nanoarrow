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
  ipc_file=$(echo "${json_gz_file}" | sed -e s/.json.gz/.stream/)
  json_gz_label=$(basename ${json_gz_file})
  ipc_label=$(basename ${ipc_file})

  gzip --decompress -c "${json_gz_file}" | \
    ./integration_test_util \
      --from ipc "${ipc_file}" \
      --check json -

  if [ $? -eq 0 ]; then
    echo "[v] ${json_gz_label} --check ${ipc_label}"
  else
    echo "[X] ${json_gz_label} --check ${ipc_label}"
    N_FAIL=$((N_FAIL+1))
  fi
done

exit $N_FAIL
