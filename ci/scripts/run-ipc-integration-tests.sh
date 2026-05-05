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
# test files from the arrow-testing repository.
#
# Usage:
#   export NANOARROW_ARROW_TESTING_DIR=/path/to/arrow-testing
#   ./dev/run_ipc_integration_tests.sh [build_dir]
#
# Arguments:
#   build_dir: Optional path to the build directory containing
#              nanoarrow_ipc_integration. Defaults to "build".

set -e

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

# Function to run VALIDATE for a given test file
run_validate() {
    local subdir="$1"
    local basename="$2"

    local stream_file="${DATA_DIR}/${subdir}/${basename}.stream"
    local arrow_file="${DATA_DIR}/${subdir}/${basename}.arrow_file"
    local json_gz="${DATA_DIR}/${subdir}/${basename}.json.gz"
    local json_file="${TEMP_DIR}/${subdir}_${basename}.json"

    # The VALIDATE command uses FromIpcFile which requires Arrow file format
    # (with ARROW1 magic and footer), not IPC stream format.
    # So we prefer .arrow_file over .stream
    local arrow_path=""
    if [ -f "${arrow_file}" ]; then
        arrow_path="${arrow_file}"
    elif [ -f "${stream_file}" ]; then
        # Note: .stream files may fail because FromIpcFile expects file format
        arrow_path="${stream_file}"
    else
        echo "SKIP: ${subdir}/${basename} - no .arrow_file or .stream found"
        return 0
    fi

    # Check if JSON exists (possibly gzipped)
    if [ -f "${json_gz}" ]; then
        gunzip -c "${json_gz}" > "${json_file}"
    elif [ -f "${DATA_DIR}/${subdir}/${basename}.json" ]; then
        json_file="${DATA_DIR}/${subdir}/${basename}.json"
    else
        echo "SKIP: ${subdir}/${basename} - no .json or .json.gz found"
        return 0
    fi

    echo "Testing: ${subdir}/${basename}"
    if COMMAND=VALIDATE ARROW_PATH="${arrow_path}" JSON_PATH="${json_file}" "${INTEGRATION_BIN}"; then
        echo "  PASS"
        return 0
    else
        echo "  FAIL"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
SKIPPED=0

run_test() {
    if run_validate "$@"; then
        if [[ $(run_validate "$@" 2>&1) == *"SKIP"* ]]; then
            ((SKIPPED++))
        else
            ((PASSED++))
        fi
    else
        ((FAILED++))
    fi
}

echo "=== Running IPC Integration Tests ==="
echo "Using arrow-testing at: ${NANOARROW_ARROW_TESTING_DIR}"
echo "Using integration binary at: ${INTEGRATION_BIN}"
echo ""

# Test files in cpp-21.0.0 (includes decimal32, decimal64, and dictionaries)
CPP_21_FILES=(
    "generated_decimal32"
    "generated_decimal64"
    "generated_decimal"
    "generated_decimal256"
    "generated_primitive"
    "generated_primitive_no_batches"
    "generated_primitive_zerolength"
    "generated_datetime"
    "generated_interval"
    "generated_interval_mdn"
    "generated_duration"
    "generated_nested"
    "generated_nested_large_offsets"
    "generated_null"
    "generated_null_trivial"
    "generated_custom_metadata"
    "generated_duplicate_fieldnames"
    "generated_map"
    "generated_map_non_canonical"
    "generated_recursive_nested"
    "generated_union"
    "generated_binary"
    "generated_binary_no_batches"
    "generated_binary_zerolength"
    "generated_large_binary"
    "generated_dictionary"
    "generated_dictionary_unsigned"
    "generated_nested_dictionary"
)

echo "=== Testing cpp-21.0.0 files ==="
for file in "${CPP_21_FILES[@]}"; do
    if run_validate "cpp-21.0.0" "${file}"; then
        ((PASSED++))
    else
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            ((SKIPPED++))
        else
            ((FAILED++))
        fi
    fi
done

# Test files in 1.0.0-littleendian
LITTLEENDIAN_FILES=(
    "generated_decimal"
    "generated_decimal256"
    "generated_primitive"
    "generated_datetime"
    "generated_interval"
    "generated_nested"
    "generated_null"
    "generated_custom_metadata"
    "generated_map"
    "generated_union"
)

echo ""
echo "=== Testing 1.0.0-littleendian files ==="
for file in "${LITTLEENDIAN_FILES[@]}"; do
    if run_validate "1.0.0-littleendian" "${file}"; then
        ((PASSED++))
    else
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            ((SKIPPED++))
        else
            ((FAILED++))
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Passed: ${PASSED}"
echo "Failed: ${FAILED}"
echo "Skipped: ${SKIPPED}"

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi
