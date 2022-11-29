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
#
# Requirements
# - cmake >= 3.16
# - R >= 3.5.0

set -e
set -o pipefail

if [ ${VERBOSE:-0} -gt 0 ]; then
  set -x
fi

case $# in
  0) VERSION="HEAD"
     SOURCE_KIND="local"
     TEST_BINARIES=0
     ;;
  1) VERSION="$1"
     SOURCE_KIND="git"
     TEST_BINARIES=0
     ;;
  2) VERSION="$1"
     RC_NUMBER="$2"
     SOURCE_KIND="tarball"
     ;;
  *) echo "Usage:"
     echo "  Verify release candidate:"
     echo "    $0 X.Y.Z RC_NUMBER"
     echo "  Verify only the source distribution:"
     echo "    TEST_DEFAULT=0 TEST_SOURCE=1 $0 X.Y.Z RC_NUMBER"
     echo "  Verify only the binary distributions:"
     echo "    TEST_DEFAULT=0 TEST_BINARIES=1 $0 X.Y.Z RC_NUMBER"
     echo "  Verify only the wheels:"
     echo "    TEST_DEFAULT=0 TEST_WHEELS=1 $0 X.Y.Z RC_NUMBER"
     echo ""
     echo "  Run the source verification tasks on a remote git revision:"
     echo "    $0 GIT-REF"
     echo "  Run the source verification tasks on this arrow checkout:"
     echo "    $0"
     exit 1
     ;;
esac

show_header() {
  echo ""
  printf '=%.0s' $(seq ${#1}); printf '\n'
  echo "${1}"
  printf '=%.0s' $(seq ${#1}); printf '\n'
}

show_info() {
  echo "└ ${1}"
}

NANOARROW_DIST_URL='https://dist.apache.org/repos/dist/dev/nanoarrow'

download_dist_file() {
  curl \
    --silent \
    --show-error \
    --fail \
    --location \
    --remote-name $ARROW_DIST_URL/$1
}

download_rc_file() {
  download_dist_file apache-arrow-nanoarrow-${VERSION}-rc${RC_NUMBER}/$1
}

import_gpg_keys() {
  if [ "${GPGKEYS_ALREADY_IMPORTED:-0}" -gt 0 ]; then
    return 0
  fi
  download_dist_file KEYS
  gpg --import KEYS

  GPGKEYS_ALREADY_IMPORTED=1
}

if type shasum >/dev/null 2>&1; then
  sha256_verify="shasum -a 256 -c"
  sha512_verify="shasum -a 512 -c"
else
  sha256_verify="sha256sum -c"
  sha512_verify="sha512sum -c"
fi

fetch_archive() {
  import_gpg_keys

  local dist_name=$1
  download_rc_file ${dist_name}.tar.gz
  download_rc_file ${dist_name}.tar.gz.asc
  download_rc_file ${dist_name}.tar.gz.sha256
  download_rc_file ${dist_name}.tar.gz.sha512
  gpg --verify ${dist_name}.tar.gz.asc ${dist_name}.tar.gz
  ${sha256_verify} ${dist_name}.tar.gz.sha256
  ${sha512_verify} ${dist_name}.tar.gz.sha512
}

verify_dir_artifact_signatures() {
  import_gpg_keys

  # verify the signature and the checksums of each artifact
  find $1 -name '*.asc' | while read sigfile; do
    artifact=${sigfile/.asc/}
    gpg --verify $sigfile $artifact || exit 1

    # go into the directory because the checksum files contain only the
    # basename of the artifact
    pushd $(dirname $artifact)
    base_artifact=$(basename $artifact)
    if [ -f $base_artifact.sha256 ]; then
      ${sha256_verify} $base_artifact.sha256 || exit 1
    fi
    if [ -f $base_artifact.sha512 ]; then
      ${sha512_verify} $base_artifact.sha512 || exit 1
    fi
    popd
  done
}

setup_tempdir() {
  cleanup() {
    if [ "${TEST_SUCCESS}" = "yes" ]; then
      rm -fr "${NANOARROW_TMPDIR}"
    else
      echo "Failed to verify release candidate. See ${NANOARROW_TMPDIR} for details."
    fi
  }

  show_header "Creating temporary directory"

  if [ -z "${NANOARROW_TMPDIR}" ]; then
    # clean up automatically if NANOARROW_TMPDIR is not defined
    NANOARROW_TMPDIR=$(mktemp -d -t "nanoarrow-${VERSION}.XXXXX")
    trap cleanup EXIT
  else
    # don't clean up automatically
    mkdir -p "${NANOARROW_TMPDIR}"
  fi

  echo "Working in sandbox ${NANOARROW_TMPDIR}"
}

test_and_install_c() {
  show_header "Build, install, and test C library"

  mkdir -p $NANOARROW_TMPDIR/build
  pushd $NANOARROW_TMPDIR/build

  cmake ${NANOARROW_SOURCE_DIR} -DNANOARROW_BUILD_TESTS=ON
  cmake --build .
  cmake --install . --prefix=$NANOARROW_HOME
  ctest --output-on-failure

  popd
}

test_c_bundled() {
  show_header "Build test C library"

  mkdir -p $NANOARROW_TMPDIR/build_bundled
  pushd $NANOARROW_TMPDIR/build_bundled

  cmake ${NANOARROW_SOURCE_DIR} -DNANOARROW_BUILD_TESTS=ON -DNANOARROW_BUNDLE=ON
  cmake --build .
  ctest --output-on-failure

  popd
}

test_r() {
  show_header "Build and test R package"

  if [ -z "${R_HOME}" ]; then
    R_BIN=${R_HOME}/bin/R
  else
    R_BIN=R
  fi

  $R_BIN -e 'install.packages("pak"); pak::local_install_dev_deps("r")'

  pushd r
  ./cleanup
  ./configure
  popd

  R_PACKAGE_TARBALL=`$R_BIN CMD build r`
  $R_BIN CMD install $R_PACKAGE_TARBALL
  $R_BIN -e 'testthat::test_local("r")'
  $R_BIN CMD check $R_PACKAGE_TARBALL
}

ensure_source_directory() {
  show_header "Ensuring source directory"

  dist_name="apache-arrow-nanoarrow-${VERSION}"

  if [ "${SOURCE_KIND}" = "local" ]; then
    # Local nanoarrow repository, testing repositories should be already present
    if [ -z "$ARROW_SOURCE_DIR" ]; then
      export ARROW_SOURCE_DIR="${ARROW_DIR}"
    fi
    echo "Verifying local Arrow checkout at ${ARROW_SOURCE_DIR}"
  elif [ "${SOURCE_KIND}" = "git" ]; then
    # Remote nanoarrow repository, testing repositories must be cloned
    : ${SOURCE_REPOSITORY:="https://github.com/apache/arrow-nanoarrow"}
    echo "Verifying nanoarrow repository ${SOURCE_REPOSITORY} with revision checkout ${VERSION}"
    export NANOARROW_SOURCE_DIR="${NANOARROW_TMPDIR}/arrow-nanoarrow"
    if [ ! -d "${ARROW_SOURCE_DIR}" ]; then
      git clone --recurse-submodules $SOURCE_REPOSITORY $NANOARROW_SOURCE_DIR
      git -C $NANOARROW_SOURCE_DIR checkout $VERSION
    fi
  else
    # Release tarball, testing repositories must be cloned separately
    echo "Verifying official nano release candidate ${VERSION}-rc${RC_NUMBER}"
    export NANOARROW_SOURCE_DIR="${NANOARROW_TMPDIR}/${dist_name}"
    if [ ! -d "${NANOARROW_SOURCE_DIR}" ]; then
      pushd $NANOARROW_TMPDIR
      fetch_archive ${dist_name}
      tar xf ${dist_name}.tar.gz
      popd
    fi
  fi

  # Ensure that the testing repositories are cloned
  if [ ! -d "${NANOARROW_SOURCE_DIR}/testing/data" ]; then
    git clone https://github.com/apache/arrow-testing.git ${ARROW_SOURCE_DIR}/testing
  fi

  export NANOARROW_TEST_DATA=$ARROW_SOURCE_DIR/testing/data
}


test_source_distribution() {
  export NANOARROW_HOME=$NANOARROW_TMPDIR/install
  pushd $NANOARROW_SOURCE_DIR


  test_and_install_c
  test_and_install_c_bundled

  if [ ${TEST_C} -gt 0 ]; then
    test_and_install_c
  fi

  if [ ${TEST_C_BUNDLED} -gt 0 ]; then
    test_c_bundled
  fi

  if [ ${TEST_R} -gt 0 ]; then
    test_r
  fi

  popd
}

# By default test all functionalities.
# To deactivate one test, deactivate the test and all of its dependents
# To explicitly select one test, set TEST_DEFAULT=0 TEST_X=1
: ${TEST_DEFAULT:=1}

: ${TEST_SOURCE:=${TEST_DEFAULT}}

TEST_SUCCESS=no

setup_tempdir
ensure_source_directory
test_source_distribution
test_binary_distribution

TEST_SUCCESS=yes

echo "Release candidate ${VERSION}-RC${RC_NUMBER} looks good!"
exit 0
