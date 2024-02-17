<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Release process for Apache Arrow nanoarrow

## Verifying a nanoarrow release candidate

Release candidates for nanoarrow are uploaded to https://dist.apache.org/repos/dist/dev/arrow/
prior to a release vote being called on the
[Apache Arrow developer mailing list](https://lists.apache.org/list.html?dev@arrow.apache.org).
A script (`verify-release-candidate.sh`) is provided to verify such a release candidate.
For example, to verify nanoarrow 0.4.0-rc0, one could run:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git arrow-nanoarrow
cd arrow-nanoarrow/dev/release
./verify-release-candidate.sh 0.4.0 0
```

Full verification requires [CMake](https://cmake.org/download/) to build and run the test
suite. The test suite currently depends on an Arrow C++ installation that is discoverable
by CMake (e.g., using one of the methods described in the
[Arrow installation instructions](https://arrow.apache.org/install/)). For environments
where binary packages are not provided, building and installing Arrow C++ from source
may be required. You can provide the `NANOARROW_CMAKE_OPTIONS` environment variable to
pass extra arguments to `cmake` (e.g., `-DArrow_DIR=<path/to/arrow>/lib/cmake/Arrow` or
`-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`).

Verification of the R package requires an
[R installation](https://cloud.r-project.org/) and a C/C++ compiler (e.g.,
[RTools](https://cloud.r-project.org/bin/windows/Rtools/) on Windows or XCode Command
Line Tools). You can set the `R_HOME` environment variable or
`export PATH="$PATH:/path/to/R"` (where `$R_HOME/bin/R` is the R executable)
to point to a specific R installation.

The verification script itself is written in `bash` and requires the `curl`, `gpg`, and
`shasum`/`sha512sum` commands. These are typically available from a package
manager except on Windows (see below).

To run only C library verification (requires CMake and Arrow C++ but not R or Python):

```bash
TEST_DEFAULT=0 TEST_C=1 TEST_C_BUNDLED=1 ./verify-release-candidate.sh 0.4.0 0
```

To run only R package verification (requires R but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_R=1 ./verify-release-candidate.sh 0.4.0 0
```

To run only Python verification (requires Python but not CMake or Arrow C++):

```bash
TEST_DEFAULT=0 TEST_PYTHON=1 ./verify-release-candidate.sh 0.4.0 0
```

### MacOS

On MacOS you can install all requirements except R using [Homebrew](https://brew.sh):

```bash
brew install cmake gnupg apache-arrow
```

For older MacOS or MacOS without Homebrew, you will have to install the XCode
Command Line Tools (i.e., `xcode-select --install`),
[install GnuPG](https://gnupg.org/download/),
[install CMake](https://cmake.org/download/), and build Arrow C++ from source.

```bash
# Download + build Arrow C++
curl https://github.com/apache/arrow/archive/refs/tags/apache-arrow-14.0.2.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-14.0.2/cpp \
    -DARROW_JEMALLOC=OFF -DARROW_SIMD_LEVEL=NONE \
    # Required for Arrow on old MacOS
    -DCMAKE_CXX_FLAGS="-D_LIBCPP_DISABLE_AVAILABILITY" \
    -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow
cd ..

# Pass location of install to the release verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd)/arrow/lib/cmake/Arrow -DCMAKE_CXX_FLAGS=-D_LIBCPP_DISABLE_AVAILABILITY"
```

You can install R using the instructions provided on the
[R Project Download page](https://cloud.r-project.org/bin/macosx/).

The system `python3` provided by MacOS is sufficient to verify the release
candidate.

### Conda (Linux and MacOS)

Using `conda`, one can install all requirements needed for verification on Linux
or MacOS. Users are recommended to install `gnupg` using
a system installer because of interactions with other installations that
may cause a crash.

```bash
conda create --name nanoarrow-verify-rc
conda activate nanoarrow-verify-rc
conda config --set channel_priority strict

conda install -c conda-forge compilers git cmake arrow-cpp
# For R (see below about potential interactions with system R
# before installing via conda on MacOS)
conda install -c conda-forge r-testthat r-hms r-blob r-pkgbuild r-bit64
```

Note that using conda-provided R when there is also a system install of R
on MacOS is unlikely to work.

Linux users that have built and installed a custom build of Arrow C++ may
have to `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib` before running the
verification script.

### Windows

On Windows, prerequisites can be installed using officially provided
installers:
[Visual Studio](https://visualstudio.microsoft.com/vs/),
[CMake](https://cmake.org/download/), and
[Git](https://git-scm.com/downloads) should provide the prerequisties
to verify the C library; R and Rtools can be installed using the
[official R-project installer](https://cloud.r-project.org/bin/windows/).
Arrow C++ can be built from source. The version of bash provided with
Git for Windows can be used to execute the Arrow C++ build commands and
the verification script.

```bash
# Build Arrow C++ from source
curl https://github.com/apache/arrow/archive/refs/tags/apache-arrow-14.0.2.tar.gz | \
  tar -zxf -
mkdir arrow-build && cd arrow-build
cmake ../apache-arrow-14.0.2/cpp -DCMAKE_INSTALL_PREFIX=../arrow
cmake --build .
cmake --install . --prefix=../arrow --config=Debug
cd ..

# Pass location of Arrow and R to the verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd -W)/arrow/lib/cmake/Arrow -Dgtest_force_shared_crt=ON -DNANOARROW_ARROW_STATIC=ON"
export R_HOME="/c/Program Files/R/R-4.2.2"
```

### Debian/Ubuntu

On Debian/Ubuntu (e.g., `docker run --rm -it ubuntu:latest`) you can install prerequisites using `apt`.

```bash
apt-get update && apt-get install -y git g++ cmake r-base gnupg curl python3-dev python3-venv

# For Arrow C++
apt-get install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt-get install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt-get update
apt-get install -y -V libarrow-dev
```

If you have never installed an R package before, R verification will fail when it
tries to install any missing dependencies. Because of how R is configured by
default, you must install your first package in an interactive session and select
`yes` when it asks if you would like to create a user-specific directory.

### Fedora

On recent Fedora (e.g., `docker run --rm -it fedora:latest`), you can install all prerequisites
using `dnf`:

```bash
dnf install -y git cmake R gnupg curl libarrow-devel python3-devel python3-virtualenv
```

### Arch Linux

On Arch Linux (e.g., `docker run --rm -it archlinux:latest`, you can install all prerequisites
using `pacman`):

```bash
pacman -S git gcc make cmake r-base gnupg curl arrow
```

### Alpine Linux

On Alpine Linux (e.g., `docker run --rm -it alpine:latest`), most prerequisites are available using `apk add` except for Arrow C++ which requires enabling the
community repository.

```bash
# Enable community repository for Arrow C++. Alternatively, you can build Arrow C++
# from source and pass its location via NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=...".
cat > /etc/apk/repositories << EOF; $(echo)

https://dl-cdn.alpinelinux.org/alpine/v$(cut -d'.' -f1,2 /etc/alpine-release)/main/
https://dl-cdn.alpinelinux.org/alpine/v$(cut -d'.' -f1,2 /etc/alpine-release)/community/
https://dl-cdn.alpinelinux.org/alpine/edge/testing/

EOF
apk update

apk add bash linux-headers git cmake R R-dev g++ gnupg curl apache-arrow-dev \
  python3-dev
```

### Centos7

On Centos7 (e.g., `docker run --rm -it centos:7`), most prerequisites are
available via `yum install` except Arrow C++, which must be built from
source. Arrow C++ 9.0.0 was the last version to support the default system
compiler (gcc 4.8).

```bash
yum install epel-release # needed to install R
yum install git gnupg curl R gcc-c++ cmake3

# Needed to get a warning-free R CMD check if the en_US.UTF-8 locale is not defined
# (e.g., in the centos:7 docker image)
# localedef -c -f UTF-8 -i en_US en_US.UTF-8
# export LC_ALL=en_US.UTF-8

# Build Arrow C++ 9.0.0 from source
curl -L https://github.com/apache/arrow/archive/refs/tags/apache-arrow-9.0.0.tar.gz | tar -zxf - && \
    mkdir /arrow-build && \
    cd /arrow-build && \
    cmake3 ../arrow-apache-arrow-9.0.0/cpp \
        -DARROW_JEMALLOC=OFF \
        -DARROW_SIMD_LEVEL=NONE \
        -DCMAKE_INSTALL_PREFIX=../arrow && \
    cmake3 --build . && \
    make install

# Pass location of Arrow, cmake, and ctest to the verification script
export NANOARROW_CMAKE_OPTIONS="-DArrow_DIR=$(pwd)/arrow/lib64/cmake/arrow"
export CMAKE_BIN=cmake3
export CTEST_BIN=ctest3

# gpg on centos7 errors for some keys in the Arrow KEYS file. This does
# not skip verifying signatures, just allows errors for unsupported entries in
# the global Arrow KEYS file.
export NANOARROW_ACCEPT_IMPORT_GPG_KEYS_ERROR=1

# System Python on centos7 is not new enough to support the Python package
export TEST_PYTHON=0
```

### Big endian

One can verify a nanoarrow release candidate on big endian by setting
`DOCKER_DEFAULT_PLATFORM=linux/s390x` and following the instructions for
[Alpine Linux](#alpine-linux) or [Fedora](#fedora).

## Creating a release candidate

The first step to creating a nanoarrow release is to create a `maint-VERSION` branch
(e.g., `usethis::pr_init("maint-0.4.0")`) and push the branch to `upstream`. This is
a good opportunity to run though the above instructions to make sure the verification
script and instructions are up-to-date. You may also wish to start a manual dispatch
of the [Verification workflow](https://github.com/apache/arrow-nanoarrow/actions/workflows/verify.yaml)
targeting the maint-XX branch that was just pushed.
When this is complete, run
[01-prepare.R](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/01-prepare.sh):

```bash
# from the repository root
# 01-prepare.sh <nanoarrow-dir> <prev_veresion> <version> <next_version> <rc-num>
dev/release/01-prepare.sh . 0.3.0 0.4.0 0.5.0 0
```

This will update version numbers, the changelong, and create the git tag
`apache-arrow-nanoarrow-0.4.0-rc0`. Check to make sure that the changelog
and versions are what you expect them to be before pushing the tag (you
may wish to do this by opening a dummy PR to run CI and look at the diff
from the main branch). When you are satisfied that the code at this tag
is release-candidate worthy, `git push` the tag to the `upstream` repository
(or whatever your remote name is for the `apache/arrow-nanoarrow` repo).
This will kick off a
[packaging workflow](https://github.com/apache/arrow-nanoarrow/blob/main/.github/workflows/packaging.yaml)
that will create a GitHub release and upload assets that are required for
later steps. This step can be done by any Arrow committer.

Next, all assets need to be signed by somebody whose GPG key is listed in the
[Arrow developers KEYS file](https://dist.apache.org/repos/dist/dev/arrow/KEYS)
by calling
[02-sign.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/02-sign.sh)
The caller of the script does not need to be on any particular branch to call
the script but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `GPG_KEY_ID` environment variable.

```bash
# 02-sign.sh <version> <rc-num>
dev/release/02-sign.sh 0.4.0 0
```

Finally, run
[03-source.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/03-source.sh).
This step can be done by any Arrow committer. The caller of this script does not need to
be on any particular branch but *does* need the
[dev/release/.env](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/.env.example)
file to exist setting the appropriate `APACHE_USERNAME` environment variable.

```
# 03-source.sh $0 <version> <rc-num>
dev/release/03-source.sh 0.4.0 0
```

You should check that the release verification runs locally and/or
start a
[Verification workflow](https://github.com/apache/arrow-nanoarrow/actions/workflows/verify.yaml) and wait for it to complete.

At this point the release candidate is suitable for a vote on the Apache Arrow developer mailing list.

```
Hello,

I would like to propose the following release candidate (rc0) of Apache Arrow nanoarrow [0] version 0.4.0. This is an initial release consisting of 44 resolved GitHub issues from 5 contributors [1].

This release candidate is based on commit: {rc_commit} [2]

The source release rc0 is hosted at [3].
The changelog is located at [4].

Please download, verify checksums and signatures, run the unit tests, and vote on the release. See [5] for how to validate a release candidate.

The vote will be open for at least 72 hours.

[ ] +1 Release this as Apache Arrow nanoarrow 0.4.0
[ ] +0
[ ] -1 Do not release this as Apache Arrow nanoarrow 0.4.0 because...

[0] https://github.com/apache/arrow-nanoarrow
[1] https://github.com/apache/arrow-nanoarrow/milestone/4?closed=1
[2] https://github.com/apache/arrow-nanoarrow/tree/apache-arrow-nanoarrow-0.4.0-rc0
[3] https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-nanoarrow-0.4.0-rc0/
[4] https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-0.4.0-rc0/CHANGELOG.md
[5] https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/README.md
```

## Post-release

After a passing release vote, the following tasks must be completed:

```
[ ] Closed GitHub milestone
[ ] Added release to the Apache Reporter System
[ ] Uploaded artifacts to Subversion
[ ] Created GitHub release
[ ] Submit R package to CRAN
[ ] Submit Python package to PyPI
[ ] Update Python package on conda-forge
[ ] Release blog post at https://github.com/apache/arrow-site/pull/288
[ ] Sent announcement to announce@apache.org
[ ] Removed old artifacts from SVN
[ ] Bumped versions on main
```

### Close GitHub milestone

Find the appropriate entry in https://github.com/apache/arrow-nanoarrow/milestones/
and mark it as closed.

### Add release to the Apache Reporter System

The reporter system for Arrow can be found at
<https://reporter.apache.org/addrelease.html?arrow>. To add a release, a
PMC member must log in with their Apache username/password. The release
names are in the form `NANOARROW-0.4.0`.

### Upload artifacts to Subversion / Create GitHub Release

These are both handled by
[post-01-upload.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/post-01-upload.sh).
This script must be run by a PMC member whose `APACHE_USERNAME` environment variable
has been set in `.env`.

```bash
dev/release/post-01-upload.sh 0.4.0 0
```

### Submit R package to CRAN

The R package submission occurs from a separate branch to facilitate including
any small changes requested by a member of the CRAN team; however, these
updates are usually automatic and do not require additional changes.
Before a release candidate is created, the first section of
`usethis::use_release_issue()` should all be completed (i.e., any changes
after release should be minor tweaks). The steps are:

- Ensure you are on the release branch (i.e., `git switch maint-0.4.0`)
- Run `usethis::pr_init("r-cran-maint-0.4.0")` and push the branch to your
  fork.
- Ensure `cran_comments.md` is up-to-date.
- Run `devtools::check()` locally and verify that the package version is correct
- Run `urlchecker::url_check()`
- Run `devtools::check_win_devel()` and wait for the response
- Run `devtools::submit_cran()`
- Confirm submission email

Any changes required at this stage should be made as a PR into `main` and
cherry-picked into the `r-cran-maint-XXX` packaging branch. (i.e.,
`git cherry-pick 01234abcdef`). If any changes
to the source are required, bump the "tweak" version (e.g., `Version: 0.4.0.1`
in `DESCRIPTION`).

### Release blog post

Final review + merge of the blog post that was drafted prior to preparation of
the release candidate.

### Send announcement

This email should be sent to announce@apache.org and dev@arrow.apache.org. It
**must** be sent from your Apache email address and **must** be sent through
the `mail-relay.apache.org` outgoing server.

Email template:

```
[ANNOUNCE] Apache Arrow nanoarrow 0.4.0 Released

The Apache Arrow community is pleased to announce the 0.4.0 release of Apache Arrow nanoarrow. This initial release covers 44 resolved issues from 5 contributors[1].

The release is available now from [2].

Release notes are available at:
https://github.com/apache/arrow-nanoarrow/blob/apache-arrow-nanoarrow-0.4.0/CHANGELOG.md

What is Apache Arrow?
---------------------
Apache Arrow is a columnar in-memory analytics layer designed to accelerate big data. It houses a set of canonical in-memory representations of flat and hierarchical data along with multiple language-bindings for structure manipulation. It also provides low-overhead streaming and batch messaging, zero-copy interprocess communication (IPC), and vectorized in-memory analytics libraries. Languages currently supported include C, C++, C#, Go, Java, JavaScript, Julia, MATLAB, Python, R, Ruby, and Rust.

What is Apache Arrow nanoarrow?
--------------------------
Apache Arrow nanoarrow is a C library for building and interpreting Arrow C Data interface structures with bindings for users of R and Python. The vision of nanoarrow is that it should be trivial for a library or application to implement an Arrow-based interface. The library provides helpers to create types, schemas, and metadata, an API for building arrays element-wise,
and an API to extract elements element-wise from an array. For a more detailed description of the features nanoarrow provides and motivation for its development, see [3].

Please report any feedback to the mailing lists ([4], [5]).

Regards,
The Apache Arrow Community

[1]: https://github.com/apache/arrow-nanoarrow/issues?q=is%3Aissue+milestone%3A%22nanoarrow+0.4.0%22+is%3Aclosed
[2]: https://www.apache.org/dyn/closer.cgi/arrow/apache-arrow-nanoarrow-0.4.0
[3]: https://github.com/apache/arrow-nanoarrow
[4]: https://lists.apache.org/list.html?user@arrow.apache.org
[5]: https://lists.apache.org/list.html?dev@arrow.apache.org
```

### Remove old artifacts from SVN

These artifacts include any release candidates that were uploaded to
<https://dist.apache.org/repos/dist/dev/arrow/>. You can remove them
using:

```
# Once
export APACHE_USERNAME=xxx
# Once for every release candidate
svn rm --username=$APACHE_USERNAME -m "Clean up svn artifacts" https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-nanoarrow-0.4.0-rc0/
```

### Bumped versions on main

This is handled by
[post-02-bump-versions.sh](https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/post-02-bump-versions.sh). Create a branch and then run:

```bash
dev/release/post-02-bump-versions.sh . 0.4.0 0.5.0
```
