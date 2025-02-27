#!/usr/bin/env bash
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

set -eu

main() {
    local -r source_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    local -r source_top_dir="$( cd "${source_dir}/../../" && pwd )"
    pushd "${source_top_dir}"

    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 <version> <rc-num>"
        exit 1
    fi

    local -r version="$1"
    local -r rc_number="$2"
    local -r tag="apache-arrow-nanoarrow-${version}-rc${rc_number}"
    local -r tarball="apache-arrow-nanoarrow-${version}"

    : ${REPOSITORY:="apache/arrow-nanoarrow"}

    if [[ ! -f "${source_dir}/.env" ]]; then
        echo "You must create ${source_dir}/.env"
        echo "You can use ${source_dir}/.env.example as a template"
    fi

    source "${source_dir}/.env"

    header "Looking for GitHub Actions workflow on ${REPOSITORY}:${tag}"
    local run_id=""
    while [[ -z "${run_id}" ]]
    do
        echo "Waiting for run to start..."
        run_id=$(gh run list \
                    --repo "${REPOSITORY}" \
                    --workflow=packaging.yaml \
                    --json 'databaseId,event,headBranch,status' \
                    --jq ".[] | select(.event == \"push\" and .headBranch == \"${tag}\") | .databaseId" | \
                    head -n 1)
        sleep 1
    done

    header "Found GitHub Actions workflow with ID: ${run_id}"
    gh run watch --repo "${REPOSITORY}" --exit-status ${run_id}
    gh run view --repo "${REPOSITORY}" "${run_id}"

    header "Downloading assets from release"
    local -r download_dir="packages/${tag}"
    mkdir -p "${download_dir}"
    gh release download \
       "${tag}" \
       --repo "${REPOSITORY}" \
       --dir "${download_dir}" \
       --skip-existing

    header "Adding release notes"
    local -r release_notes=$(python3 ${source_dir}/release_tools.py changelog)
    echo "${release_notes}"
    gh release edit \
       "${tag}" \
       --repo "${REPOSITORY}" \
       --notes "${release_notes}"

    header "Upload signatures for source"
    upload_asset_signatures "${tag}" $(find "${download_dir}" -type f \( -name 'apache-arrow-nanoarrow-*.tar.gz' \))

    header "Upload signatures for docs"
    upload_asset_signatures "${tag}" "${download_dir}/docs.tgz"

    popd
}

header() {
    echo "============================================================"
    echo "${1}"
    echo "============================================================"
}

sign_asset() {
    local -r asset="$1"
    local -r sigfile="${asset}.asc"

    if [[ -f "${sigfile}" ]]; then
        if env LANG=C gpg --verify "${sigfile}" "${asset}" >/dev/null 2>&1; then
            echo "Valid signature at $(basename "${sigfile}"), skipping"
            return
        fi
        rm "${sigfile}"
    fi

    gpg \
        --armor \
        --detach-sign \
        --local-user "${GPG_KEY_ID}" \
        --output "${sigfile}" \
        "${asset}"
    echo "Generated $(basename "${sigfile}")"
}

sum_asset() {
    local -r asset="$1"
    local -r sumfile="${asset}.sha512"

    local -r digest=$(cd $(dirname "${asset}"); shasum --algorithm 512 $(basename "${asset}"))
    if [[ -f "${sumfile}" ]]; then
        if [[ "${digest}" = $(cat "${sumfile}") ]]; then
            echo "Valid digest at $(basename "${sumfile}"), skipping"
            return
        fi
    fi

    echo "${digest}" > "${sumfile}"
    echo "Generated $(basename "${sumfile}")"
}

upload_asset_signatures() {
    local -r tag="${1}"
    shift 1

    local -r assets=("$@")

    for asset in "${assets[@]}"; do
        sign_asset "${asset}"
        sum_asset "${asset}"
    done

    gh release upload \
       --repo "${REPOSITORY}" \
       "${tag}" \
       "${assets[@]/%/.asc}" \
       "${assets[@]/%/.sha512}"

    # Clean up
    for asset in "${assets[@]}"; do
        rm -f "${asset}" "${asset}.asc" "${asset}.sha512"
    done
}

main "$@"
