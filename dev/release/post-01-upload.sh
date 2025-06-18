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

    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 <version> <rc-num>"
        exit 1
    fi
    local -r version="$1"
    local -r rc_number="$2"
    local -r tag="apache-arrow-nanoarrow-${version}-rc${rc_number}"

    if [[ ! -f "${source_dir}/.env" ]]; then
        echo "You must create ${source_dir}/.env"
        echo "You can use ${source_dir}/.env.example as a template"
    fi

    source "${source_dir}/.env"

    rc_id="apache-arrow-nanoarrow-${version}-rc${rc_number}"
    release_id="apache-arrow-nanoarrow-${version}"
    echo "Moving dev/ to release/"
    svn \
        mv \
        --username=$APACHE_USERNAME \
        -m "Apache Arrow nanoarrow ${version}" \
        https://dist.apache.org/repos/dist/dev/arrow/${rc_id} \
        https://dist.apache.org/repos/dist/release/arrow/${release_id}

    echo "Create final tag"
    git tag -a "apache-arrow-nanoarrow-${version}" -m "nanoarrow ${version}" "${tag}^{}"

    echo "Success! The release is available here:"
    echo "  https://dist.apache.org/repos/dist/release/arrow/${release_id}"
    echo "git push upstream apache-arrow-nanoarrow-${version} before continuing!"
}

main "$@"
