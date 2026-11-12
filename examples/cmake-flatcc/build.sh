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

set -exuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../.." && pwd)"
root="$(mktemp -d "${TMPDIR:-/tmp}/nanoarrow-flatcc-XXXXXX")"
trap 'rm -rf "${root}"' EXIT

# CMake needs native paths in cache variables when running from Git Bash.
cmake_path() {
    if [[ "${OSTYPE}" == msys* || "${OSTYPE}" == cygwin* ]]; then
        cygpath -m "$1"
    else
        printf '%s\n' "$1"
    fi
}

source_dir="$(cmake_path "${source_dir}")"
layouts=(static shared)
# On Windows static and import libraries have the same filename.
if [[ "${OSTYPE}" != msys* && "${OSTYPE}" != cygwin* ]]; then
    layouts+=(both)
fi

for layout in "${layouts[@]}"; do
    prefix="${root}/${layout}"
    postfix="_d"
    expect_static=ON
    case "${layout}" in
        static) types=(STATIC) ;;
        shared)
            types=(SHARED)
            postfix=""
            expect_static=OFF
            ;;
        both) types=(SHARED STATIC) ;;
    esac

    for kind in "${types[@]}"; do
        for config in Release Debug; do
            build="${root}/fixture-${layout}-${kind}-${config}"
            cmake -S "$(cmake_path "${script_dir}/fixture")" -B "$(cmake_path "${build}")" \
                "-DNANOARROW_SOURCE=${source_dir}" \
                "-DCMAKE_BUILD_TYPE=${config}" \
                "-DRUNTIME_TYPE=${kind}" \
                "-DRUNTIME_DEBUG_POSTFIX=${postfix}" \
                "-DCMAKE_INSTALL_PREFIX=$(cmake_path "${prefix}")"
            cmake --build "$(cmake_path "${build}")" --config "${config}" --parallel 2
            cmake --install "$(cmake_path "${build}")" --config "${config}"
        done
    done

    for mode in root dirs; do
        for config in Release Debug; do
            build="${root}/consumer-${layout}-${mode}-${config}"
            libdir="${prefix}/lib"
            bindir="${prefix}/bin"
            if [ "${config}" = Debug ]; then
                libdir="${prefix}/debug/lib"
                bindir="${prefix}/debug/bin"
            fi

            options=("-DNANOARROW_FLATCC_ROOT_DIR=$(cmake_path "${prefix}")")
            expect_debug=ON
            if [ "${mode}" = dirs ]; then
                # Separate-directory discovery uses the unsuffixed name.
                if [ "${config}" = Debug ] && [ -n "${postfix}" ]; then
                    continue
                fi
                options=(
                    "-DNANOARROW_FLATCC_INCLUDE_DIR=$(cmake_path "${prefix}/include")"
                    "-DNANOARROW_FLATCC_LIB_DIR=$(cmake_path "${libdir}")"
                )
                expect_debug=OFF
            fi

            cmake -S "$(cmake_path "${script_dir}/consumer")" -B "$(cmake_path "${build}")" \
                "-DNANOARROW_SOURCE=${source_dir}" \
                "-DCMAKE_BUILD_TYPE=${config}" \
                "-DEXPECT_DEBUG=${expect_debug}" \
                "-DEXPECT_STATIC=${expect_static}" \
                "${options[@]}"
            cmake --build "$(cmake_path "${build}")" --config "${config}" \
                --target consumer nanoarrow_ipc_shared --parallel 2
            # --test-dir was added after the supported CMake minimum.
            (
                cd "${build}"
                PATH="${bindir}:${PATH}" ctest -C "${config}" --output-on-failure
            )
        done
    done
done

mkdir "${root}/missing"
if cmake -S "${source_dir}" -B "$(cmake_path "${root}/missing-build")" \
    -DNANOARROW_IPC=ON \
    "-DNANOARROW_FLATCC_INCLUDE_DIR=${source_dir}/thirdparty/flatcc/include" \
    "-DNANOARROW_FLATCC_LIB_DIR=$(cmake_path "${root}/missing")" \
    > "${root}/missing.log" 2>&1; then
    cat "${root}/missing.log"
    echo "Expected missing-runtime configuration failure" >&2
    exit 1
fi
if ! grep -q NANOARROW_FLATCC_LIBRARY "${root}/missing.log"; then
    cat "${root}/missing.log"
    echo "Configuration failed for a reason other than a missing runtime" >&2
    exit 1
fi

echo "External flatcc discovery checks passed."
