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

NANOARROW_DIR="${SOURCE_DIR}/../.."

update_versions() {
  local base_version=$1
  local next_version=$2
  local type=$3

  case ${type} in
    release)
      local version=${base_version}
      local docs_version=${base_version}
      local python_version=${base_version}
      local r_version=${base_version}
      ;;
    snapshot)
      local version=${next_version}-SNAPSHOT
      local docs_version="${next_version} (dev)"
      local python_version="${next_version}.dev0"
      local r_version="${base_version}.9000"
      ;;
  esac
  local major_version=${version%%.*}

  pushd "${NANOARROW_DIR}"
  sed -i.bak -E "s/set\(NANOARROW_VERSION \".+\"\)/set(NANOARROW_VERSION \"${version}\")/g" CMakeLists.txt
  rm CMakeLists.txt.bak
  git add CMakeLists.txt
  sed -i.bak -E "s/version: '.+'\/version: '${version}')/g" meson.build
  rm meson.build.bak
  git add meson.build
  popd

  pushd "${NANOARROW_DIR}/r"
  Rscript -e "desc::desc_set(Version = '${r_version}')"
  git add DESCRIPTION
  popd

  pushd "${NANOARROW_DIR}/python/src/nanoarrow"
  sed -i.bak -E "s/version = \".+\"/version = \"${python_version}\"/" _static_version.py
  rm _static_version.py.bak
  git add _static_version.py
  popd
}

set_resolved_issues() {
    # TODO: this needs to work with open milestones
    local -r version="${1}"
    local -r milestone_info=$(gh api \
                                 /repos/apache/arrow-nanoarrow/milestones \
                                 -X GET \
                                 -F state=all \
                                 --jq ".[] | select(.title | test(\"nanoarrow ${version}\$\"))")
    local -r milestone_number=$(echo "${milestone_info}" | jq -r '.number')

    local -r graphql_query="query {
    repository(owner: \"apache\", name: \"arrow-nanoarrow\") {
        milestone(number: ${milestone_number}) {
            issues(states: CLOSED) {
                totalCount
            }
        }
    }
}
"

    export MILESTONE_URL=$(echo "${milestone_info}" | jq -r '.html_url')
    export RESOLVED_ISSUES=$(gh api graphql \
                            -f query="${graphql_query}" \
                            --jq '.data.repository.milestone.issues.totalCount')
}
