#!/bin/bash

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

# This script runs the nanoarrow_ipc_integration VALIDATE command against
# all test files from the arrow-testing repository's integration directory.
#
# Usage:
#   export NANOARROW_ARROW_TESTING_DIR=/path/to/arrow-testing
#   ./ci/scripts/run-ipc-integration-tests.sh [build_dir]
#
# Arguments:
#   build_dir: Optional path to the build directory containing
#              nanoarrow_ipc_integration. Defaults to "build".

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build}"

if [ -z "${NANOARROW_ARROW_TESTING_DIR}" ]; then
    echo "Error: NANOARROW_ARROW_TESTING_DIR environment variable not set"
    echo "Please set it to the path of a checkout of apache/arrow-testing"
    exit 1
fi

if [ ! -d "${NANOARROW_ARROW_TESTING_DIR}" ]; then
    echo "Error: NANOARROW_ARROW_TESTING_DIR does not exist: ${NANOARROW_ARROW_TESTING_DIR}"
    exit 1
fi

INTEGRATION_BIN="${BUILD_DIR}/nanoarrow_ipc_integration"
if [ ! -x "${INTEGRATION_BIN}" ]; then
    echo "Error: nanoarrow_ipc_integration not found at ${INTEGRATION_BIN}"
    echo "Please build the project first or specify the build directory as an argument"
    exit 1
fi

DATA_DIR="${NANOARROW_ARROW_TESTING_DIR}/data/arrow-ipc-stream/integration"

# Create a temp directory for decompressed JSON files
TEMP_DIR=$(mktemp -d)
trap "rm -rf ${TEMP_DIR}" EXIT

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Known files that are expected to be skipped (unsupported types)
SKIP_PATTERNS=(
    "generated_list_view"      # ListView not supported
    "generated_binary_view"    # BinaryView not supported
    "generated_run_end_encoded" # REE not supported
)

# Function to check if a file should be skipped
should_skip() {
    local basename="$1"
    for pattern in "${SKIP_PATTERNS[@]}"; do
        if [[ "${basename}" == *"${pattern}"* ]]; then
            return 0
        fi
    done
    return 1
}

# Function to run VALIDATE for a given test file
run_validate() {
    local subdir="$1"
    local basename="$2"

    local arrow_file="${DATA_DIR}/${subdir}/${basename}.arrow_file"
    local json_gz="${DATA_DIR}/${subdir}/${basename}.json.gz"
    local json_file="${TEMP_DIR}/${subdir}_${basename}.json"

    # We require .arrow_file format (with ARROW1 magic and footer)
    if [ ! -f "${arrow_file}" ]; then
        return 2  # Skip - no arrow file
    fi

    # Check if JSON exists (possibly gzipped)
    if [ -f "${json_gz}" ]; then
        gunzip -c "${json_gz}" > "${json_file}"
    elif [ -f "${DATA_DIR}/${subdir}/${basename}.json" ]; then
        json_file="${DATA_DIR}/${subdir}/${basename}.json"
    else
        return 2  # Skip - no JSON file
    fi

    if COMMAND=VALIDATE ARROW_PATH="${arrow_file}" JSON_PATH="${json_file}" "${INTEGRATION_BIN}" > /dev/null 2>&1; then
        return 0  # Pass
    else
        return 1  # Fail
    fi
}

echo "=== Running IPC Integration Tests ==="
echo "Using arrow-testing at: ${NANOARROW_ARROW_TESTING_DIR}"
echo "Using integration binary at: ${INTEGRATION_BIN}"
echo ""

# Find all subdirectories in the integration directory
for subdir_path in "${DATA_DIR}"/*/; do
    [ -d "${subdir_path}" ] || continue
    subdir=$(basename "${subdir_path}")

    # Skip versions before 1.0.0
    if [[ "${subdir}" == 0.* ]]; then
        continue
    fi

    echo "=== Testing ${subdir} ==="

    # Find all unique basenames (from .arrow_file files)
    for arrow_file in "${subdir_path}"*.arrow_file; do
        [ -f "${arrow_file}" ] || continue

        basename=$(basename "${arrow_file}" .arrow_file)

        # Check if this file should be skipped
        if should_skip "${basename}"; then
            ((SKIPPED++))
            continue
        fi

        run_validate "${subdir}" "${basename}"
        result=$?

        if [ $result -eq 0 ]; then
            echo "  PASS: ${basename}"
            ((PASSED++))
        elif [ $result -eq 2 ]; then
            echo "  SKIP: ${basename} (missing files)"
            ((SKIPPED++))
        else
            echo "  FAIL: ${basename}"
            ((FAILED++))
        fi
    done
    echo ""
done

echo "=== Summary ==="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo "Skipped: ${SKIPPED}"

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi
